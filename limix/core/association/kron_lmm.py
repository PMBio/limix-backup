from limix.core.type.cached import Cached, cached
from limix.core.type.observed import Observed

import sys
import numpy.linalg as la 
import numpy as np
import scipy as sp
import limix.core.association.kron_util as kron_util
import mingrid
import time

class KroneckerLMM(Cached):
	def __init__(self, Y, R1, C1, R2, C2, X, A=None, h2=0.5, reml=True):
		Cached.__init__(self)
		C = []
		R = []
		C.append(C1)
		C.append(C2)
		R.append(R1)
		R.append(R2)
		self.C = C
		self.R = R
		if type(X) is list:
			self.X = X
		else:
			self.X = [X]
		if type(A) is list:
			self.A = A
		else:
			self.A = [A]	
		assert len(self.X)==len(self.A), "missmatch between len(X)=%i and len(A)=%i" % (len(self.X), len(self.A))
		self.h2 = h2
		self.Y=Y
		self.reml=reml

	@property
	def reml(self):
		return self._reml

	@reml.setter
	def reml(self,value):
		self.clear_cache("reml")
		self._reml=value
	
	@property
	def h2(self):
		return self._h2

	@h2.setter
	def h2(self, value):
		assert value>=0.0, "h2 has to be non-negative, found %f" % value
		assert value<1.0, "h2 has to be smaller than 1.0, found %f" % valyue
		self.clear_cache("h2")
		self._h2 = value

	@property
	def Y(self):
		return self._Y

	@Y.setter
	def Y(self, value):
		self.clear_cache("Y")
		self._Y = value

	@property
	def C(self):
		return self._C

	@C.setter
	def C(self, value):
		assert len(value)==2, "length missmatch C, need 2 found %i" % len(value)
		self.clear_cache("C")
		self._C = value

	@property
	def R(self):
		return self._R

	@R.setter
	def R(self, value):
		assert len(value)==2, "length missmatch R, need 2 found %i" % len(value)
		self.clear_cache("R")
		self._R = value

	@property
	def A(self):
		return self._A

	@A.setter
	def A(self, value):
		self.clear_cache("A")
		if type(value) is list:
			self._A = value
		else:
			self._A = [value]

	@property
	def X(self):
		return self._X

	@X.setter
	def X(self, value):
		self.clear_cache("X")
		if type(value) is list:
			self._X = value
		else:
			self._X = [value]

	@property
	def P(self):
		return self.C[0].shape[0]

	@property
	def N(self):
		return self.R[0].shape[0]

	@property
	def length(self):
		assert len(self.X)==len(self.A), "missmatch between len(X)=%i and len(A)=%i" % (len(X), len(A))
		return len(self.X)
	
	def dof_X(self, X):
		if type(X) is list:
			dof = np.empty((len(X)),dtype=np.int)
			for i in xrange(len(X)):
				dof[i] = X[i].shape[1]
			return dof
		else:
			return X.shape[1]

	def dof_A(self, A):
		if type(A) is list:
			dof = np.empty((len(A)),dtype=np.int)
			for i in xrange(len(A)):
				if A[i] is None:
					dof[i] = self.P
				else:
					dof[i] = A[i].shape[0]
			return dof
		elif A is None:
			return self.P
		else:
			return A.shape[0]

	@property
	def dof(self):
		dof = self.dof_A(A=self.A) * self.dof_X(X=self.X)
		return dof

	@cached(["C","R","h2"])
	def D_rot(self):
		sR = self.Rrot()[0][0]
		sC = self.Crot()[0][0]
		return 1.0 / (self.h2 * sR[:,np.newaxis] * sC[np.newaxis,:] + (1.0-self.h2))

	def covariance_vec(self, i):
		return np.kron(self.C[i],self.R[i])

	@cached(["R"])
	def Rrot(self):
		return KroneckerLMM.rot_kron(self.R[0], self.R[1])#, h2=self.h2)

	@cached(["C"])
	def Crot(self):
		return KroneckerLMM.rot_kron(self.C[0], self.C[1])#, h2=self.h2)

	@cached(["Y","C","R"])
	def Yrot(self):
		res = self.Rrot()[0][1].T.dot(self.Y).dot(self.Crot()[0][1])
		return res

	def rotate_X(self, X):
		return self.Rrot()[0][1].T.dot(X)

	def rotate_A(self, A):
		if A is None:
			#return self.Crot()[1]
			#return np.eye(self.P)
			return None
		else:
			return A.dot(self.Crot()[0][1])

	@cached(["X","R"])
	def X_rot(self):
		res = []
		for i in xrange(self.length):
			res.append(self.rotate_X(X=self.X[i]))
		return res

	@cached(["A","C"])
	def A_rot(self):
		res = []
		for i in xrange(self.length):
			res.append(self.rotate_A(A=self.A[i]))
		return res

	@cached(["X","A","C","R","h2"])
	def XKX(self):
		dof = self.dof
		dof_cumsum = np.concatenate(([0],dof.cumsum()))
		XKX = np.empty((dof.sum(),dof.sum()))
		for i in xrange(self.length):
			XKX[dof_cumsum[i]:dof_cumsum[i+1],dof_cumsum[i]:dof_cumsum[i+1]] = kron_util.compute_X1KX2(Y=self.Y, D=self.D_rot(), A1=self.A_rot()[i], A2=self.A_rot()[i], X1=self.X_rot()[i], X2=self.X_rot()[i])
			for j in xrange(i+1,self.length):
				XKX[dof_cumsum[i]:dof_cumsum[i+1],dof_cumsum[j]:dof_cumsum[j+1]] = kron_util.compute_X1KX2(Y=self.Y, D=self.D_rot(), A1=self.A_rot()[i], A2=self.A_rot()[j], X1=self.X_rot()[i], X2=self.X_rot()[j])
				XKX[dof_cumsum[j]:dof_cumsum[j+1],dof_cumsum[i]:dof_cumsum[i+1]] = XKX[dof_cumsum[i]:dof_cumsum[i+1],dof_cumsum[j]:dof_cumsum[j+1]].T
		return XKX

	@cached(["X","A","C","R","Y","h2"])
	def XKY(self):
		dof = self.dof
		dof_cumsum = np.concatenate(([0],dof.cumsum()))
		XKY = np.empty((dof.sum(),1))
		DY = self.D_rot() * self.Yrot()
		for i in xrange(self.length):
			XKY[dof_cumsum[i]:dof_cumsum[i+1],0] = kron_util.vec(kron_util.compute_XYA(DY=DY, X=self.X_rot()[i], A=self.A_rot()[i]))		
		return XKY

	@cached(["C","R","Y","h2"])
	def YKY(self):
		YKY = (self.Yrot() * self.Yrot() * self.D_rot()).sum()
		return YKY

	@cached(["X","A","Y","C","R","h2"])
	def beta(self):
		XKX = self.XKX()
		XKy = self.XKY()
		beta = la.solve(XKX,XKy)
		return beta
	
	@cached(["X","A","Y","C","R","h2"])
	def resKres(self):
		yKy = self.YKY()
		beta = self.beta()
		XKy = self.XKY()
		rss = yKy - (beta*XKy).sum()
		return rss

	@cached(["C","R","h2"])
	def logdet_K(self):
		D = self.D_rot()
		logD = np.log(D)
		logdet_D = -logD.sum()
		logdet_R = self.P * np.log(self.Rrot()[1][0]).sum()
		logdet_C = self.N * np.log(self.Crot()[1][0]).sum()
		logdet = logdet_D + logdet_R + logdet_C
		return logdet

	@cached(["X","A","C","R","h2"])
	def logdet_XKX(self):
		XKX = self.XKX()
		sign,logdet_XKX = la.slogdet(XKX)
		return logdet_XKX

	@cached(["X","A","Y","C","R","h2","reml"])
	def LL(self):
		yKy = self.resKres()
		logdet = self.logdet_K()
		if self.reml:
			logdet += self.logdet_XKX()
			dof = self.dof.sum()
			var = yKy/(self.N*self.P-dof)
			const = (self.N*self.P-dof) * (np.log(2.0*sp.pi) + np.log(var))
		else:
			var = yKy/(self.N*self.P)
			const = self.N*self.P * (np.log(2.0*sp.pi) + np.log(var))
		dataterm = yKy/var
		logl =  -0.5 * (const + logdet + dataterm)
		return logl

	def find_h2(self, nGridH2=10, minH2=0.0, maxH2=0.99999, verbose=False):
		'''
		Finds the optimal kernel mixture weight h2 (heritability) and returns the log-likelihood
		
		(default maxA2 value is set to less than 1 as loss of positive definiteness of the final model covariance depends on h2)
		
		Args:
			nGridH2 : number of h2-grid points to evaluate the negative log-likelihood at. Number of grid points for Brent search intervals (default: 10)
			minH2   : minimum value for h2 optimization
			maxH2   : maximum value for h2 optimization
			verbose : verbose output? (default: False)
			
		Returns:
			max log likelihood obtained
		'''
		
		self.numcalls = 0
		resmin = [None]
		def f(x,resmin=resmin, **kwargs):
			self.numcalls+=1
			t0 = time.time()
			self.h2 = x
			res = -self.LL()
			
			if (resmin[0] is None) or (res < resmin[0]['nLL']):
				resmin[0] = {
				'nLL':res,
				'h2':x
				}
			t1 = time.time()
			if verbose:
				print "one objective function call took %.2f seconds" % (t1-t0)
			return res
		if verbose:
			print "findh2"
		minimum = mingrid.minimize1D(f=f, nGrid=nGridH2, minval=minH2, maxval=maxH2, verbose=verbose)
		if verbose:
			print "numcalls to log likelihood= " + str(self.numcalls)
		self.h2 = resmin[0]['h2']
		return -resmin[0]['nLL']
	
	@staticmethod
	def rot_kron(C1, C2):
		res_C2 = kron_util.pexph(C2,exp=-0.5)
		sqrtC2i = res_C2["res"]
		C1_rot = sqrtC2i.T.dot(C1).dot(sqrtC2i.T)
		s,U = la.eigh(C1_rot)
		res = (s,sqrtC2i.dot(U))
		ret = [res,res_C2["eigh"]]
		return ret

