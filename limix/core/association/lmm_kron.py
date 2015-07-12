from limix.core.type.cached import Cached, cached
from limix.core.type.observed import Observed

import sys
import ipdb
import numpy.linalg as la 
import numpy as np
import scipy as sp
import limix.core.association.kron_util as kron_util
import limix.core.fastany.fast_any as fast_any
class LmmKronecker(Cached):
	def __init__(self, Y, R1, C1, R2, C2, X, A=None, var_K1=1.0, var_K2=1.0):
		Cached.__init__(self)
		C = []
		R = []
		C.append(C1)# * var_K1)
		C.append(C2)# * var_K2)
		R.append(R1 * var_K1)
		R.append(R2 * var_K2)
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
		#self.var_total = var_K1 + var_K2
		#self.h2 = var_K1 / self.var_total
		self.Y=Y

	@property
	def C(self):
		return self._C

	@C.setter
	def C(self, value):
		assert len(value)==2, "length missmatch C, need 2 found %i" % len(value)
		self.clear_cache("default")
		self._C = value

	@property
	def R(self):
		return self._R

	@R.setter
	def R(self, value):
		assert len(value)==2, "length missmatch R, need 2 found %i" % len(value)
		self.clear_cache("default")
		self._R = value

	@property
	def A(self):
		return self._A

	@A.setter
	def A(self, value):
		self.clear_cache("fixed")
		if type(value) is list:
			self._A = value
		else:
			self._A = [value]

	@property
	def X(self):
		return self._X

	@X.setter
	def X(self, value):
		self.clear_cache("fixed")
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
	
	@property
	def dof(self):
		dof = np.empty((self.length),dtype=np.int)
		for i in xrange(self.length):
			if self.A[i] is None:
				dof[i] = self.X[i].shape[1]*self.P
			else:	
				dof[i] = self.X[i].shape[1]*self.A[i].shape[0]
		return dof

	@cached
	def D_rot(self):
		sR = self.Rrot()[0][0]
		sC = self.Crot()[0][0]
		return 1.0 / (sR[:,np.newaxis] * sC[np.newaxis,:] + 1.0)

	def covariance_vec(self, i):
		return np.kron(self.C[i],self.R[i])

	@cached
	def Rrot(self):
		return LmmKronecker.rot_kron(self.R[0], self.R[1])#, h2=self.h2)

	@cached
	def Crot(self):
		return LmmKronecker.rot_kron(self.C[0], self.C[1])#, h2=self.h2)

	@cached
	def Yrot(self):
		res = self.Rrot()[0][1].T.dot(self.Y).dot(self.Crot()[0][1])
		return res

	def X_rot_i(self, i):
		res = self.Rrot()[0][1].T.dot(self.X[i])
		return res

	@cached("fixed")
	def X_rot(self):
		res = []
		for i in xrange(self.length):
			res.append(self.X_rot_i(i=i))
		return res

	def A_rot_i(self, i):
		if self.A[i] is None:
			#return self.Crot()[1]
			#return np.eye(self.P)
			return None
		else:
			return self.A[i].dot(self.Crot()[0][1])
	
	@cached("fixed")
	def A_rot(self):
		res = []
		for i in xrange(self.length):
			res.append(self.A_rot_i(i=i))
		return res

	@cached("fixed")
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

	@cached("fixed")
	def XKY(self):
		dof = self.dof
		dof_cumsum = np.concatenate(([0],dof.cumsum()))
		XKY = np.empty((dof.sum(),1))
		DY = self.D_rot() * self.Yrot()
		for i in xrange(self.length):
			XKY[dof_cumsum[i]:dof_cumsum[i+1],0] = kron_util.vec(kron_util.compute_XYA(DY=DY, X=self.X_rot()[i], A=self.A_rot()[i]))		
		return XKY

	@cached
	def YKY(self):
		YKY = (self.Yrot() * self.Yrot() * self.D_rot()).sum()
		return YKY

	@cached("fixed")
	def beta(self):
		XKX = self.XKX()
		XKy = self.XKY()
		beta = la.solve(XKX,XKy)
		return beta
	
	@cached("fixed")
	def resKres(self):
		yKy = self.YKY()
		beta = self.beta()
		XKy = self.XKY()
		return yKy - (beta*XKy).sum()

	@cached
	def logdet_K(self):
		D = self.D_rot()
		logD = np.log(D)
		logdet_D = -logD.sum()
		logdet_R = self.P * np.log(self.Rrot()[1][0]).sum()
		logdet_C = self.N * np.log(self.Crot()[1][0]).sum()
		return logdet_D + logdet_R + logdet_C

	@cached("fixed")
	def logdet_XKX(self):
		XKX = self.XKX()
		sign,logdet_XKX = la.slogdet(XKX)
		return logdet_XKX

	def nLL(self, reml=True):
		
		yKy = self.resKres()

		logdet = self.logdet_K()

		if reml:
			logdet += self.logdet_XKX()
			dof = self.dof.sum()
			var = yKy/(self.N*self.P-dof)
			const = (self.N*self.P-dof) * np.log(2.0*sp.pi)
		else:
			var = yKy/(self.N*self.P)
			const = self.N*self.P * np.log(2.0*sp.pi)

		dataterm = yKy/var
		logl =  -0.5 * (const + logdet + dataterm)

		return logl
	
	@staticmethod
	def rot_kron(C1, C2, h2=None):
		if h2 is not None:
			raise NotImplementedError("h2 not supported yet")
			assert h2>=0.0, "h2 has to be positive, found %f" % h2
			assert h2<=1.0, "h2 has to be smaller than 1.0, found %f" % h2
			C1 = h2 * C1
			C2 = (1.0-h2) * C2
		res_C2 = kron_util.pexph(C2,exp=-0.5)
		sqrtC2i = res_C2["res"]
		C1_rot = sqrtC2i.T.dot(C1).dot(sqrtC2i.T)
		s,U = la.eigh(C1_rot)
		res = (s,sqrtC2i.dot(U))
		ret = [res,res_C2["eigh"]]
		return ret
