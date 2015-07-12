import sys
import ipdb
import numpy.linalg as la 
import numpy as np
import scipy as sp
import limix.core.association.kron_util as kron_util
import limix.core.fastany.fast_any as fast_any

class LmmKronecker(object):
	def __init__(self, Y, R1, C1, R2, C2, X, A=None, var_K1=1.0, var_K2=1.0):
		self.C = []
		self.R = []
		self.C.append(C1)# * var_K1)
		self.C.append(C2)# * var_K2)
		self.R.append(R1 * var_K1)
		self.R.append(R2 * var_K2)
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

	def D_rot(self):
		sR = self.Rrot()[0][0]
		sC = self.Crot()[0][0]
		return 1.0 / (sR[:,np.newaxis] * sC[np.newaxis,:] + 1.0)

	def covariance_vec(self, i):
		return np.kron(self.C[i],self.R[i])

	def Rrot(self):
		return LmmKronecker.rot_kron(self.R[0], self.R[1])#, h2=self.h2)

	def Crot(self):
		return LmmKronecker.rot_kron(self.C[0], self.C[1])#, h2=self.h2)

	def Yrot(self):
		res = self.Rrot()[0][1].T.dot(self.Y).dot(self.Crot()[0][1])
		return res

	def X_rot_i(self, i):
		res = self.Rrot()[0][1].T.dot(self.X[i])
		return res

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
	def A_rot(self):
		res = []
		for i in xrange(self.length):
			res.append(self.A_rot_i(i=i))
		return res

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

	def XKY(self):
		dof = self.dof
		dof_cumsum = np.concatenate(([0],dof.cumsum()))
		XKY = np.empty((dof.sum(),1))
		DY = self.D_rot() * self.Yrot()
		for i in xrange(self.length):
			XKY[dof_cumsum[i]:dof_cumsum[i+1],0] = kron_util.vec(kron_util.compute_XYA(DY=DY, X=self.X_rot()[i], A=self.A_rot()[i]))		
		return XKY

	def YKY(self):
		YKY = (self.Yrot() * self.Yrot() * self.D_rot()).sum()
		return YKY

	def Y_vec(self):
		#return self.Y.flatten(order="C")
		#return self.Y.flatten(order="F")
		result = kron_util.vec(self.Y)[:,np.newaxis]
		return result

	def beta(self):
		XKX = self.XKX()
		XKy = self.XKY()
		beta = la.solve(XKX,XKy)
		return beta
	
	def resKres(self):
		yKy = self.YKY()
		beta = self.beta()
		XKy = self.XKY()
		return yKy - (beta*XKy).sum()

	def logdet_K_vec(self):
		K = self.K_vec()
		sign,logdet_vec = la.slogdet(K)
		logdet = self.logdet_K()
		diff = logdet-logdet_vec
		if np.absolute(diff).sum()>1e-8:
			print np.absolute(diff).sum()
			import ipdb; ipdb.set_trace()
		return logdet_vec

	def logdet_K(self):
		D = self.D_rot()
		logD = np.log(D)
		logdet_D = -logD.sum()
		logdet_R = self.P * np.log(self.Rrot()[1][0]).sum()
		logdet_C = self.N * np.log(self.Crot()[1][0]).sum()
		return logdet_D + logdet_R + logdet_C

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
