import sys
import ipdb
import numpy.linalg as la 
import numpy as np
import scipy as sp
import limix.core.association.kron_util as kron_util
import limix.core.fastany.fast_any as fast_any
import limix.core.association.lmm_kron as lmm_kron

class LmmKronecker_vec(lmm_kron.LmmKronecker):
	def __init__(self, Y, R1, C1, R2, C2, X, A=None, var_K1=1.0, var_K2=1.0, diff_threshold=1e-8):
		lmm_kron.LmmKronecker.__init__(self, Y=Y, R1=R1, C1=C1, R2=R2, C2=C2, X=X, A=A, var_K1=var_K1, var_K2=var_K2)
		self.diff_threshold=diff_threshold

	def Y_vec(self):
		#return self.Y.flatten(order="C")
		#return self.Y.flatten(order="F")
		result = kron_util.vec(self.Y)[:,np.newaxis]
		return result

	def dof_vec(self):
		return self.X_vec().shape[1]

	def vecX_i(self, i):
		if self.A[i] is None:
			raise NotImplementedError("work out how A has to look, such that A_rot_i==eye(P)")
			return np.kron(la.inv(self.Crot()[0][1]),self.X[i])
		else:
			return np.kron(self.A[i],self.X[i])

	def logdet_K_vec(self):
		K = self.K_vec()
		sign,logdet_vec = la.slogdet(K)
		logdet = self.logdet_K()
		diff = logdet-logdet_vec
		if np.absolute(diff).sum()>1e-8:
			raise Exception("missmatch, diff = %f" % np.absolute(diff).sum())
		return logdet_vec

	
	def X_vec(self):
		dof = self.dof
		resX = np.empty((self.N*self.P,dof.sum()))
		dof_cumsum = np.concatenate(([0],dof.cumsum()))
		for i in xrange(self.length):
			resX[:,dof_cumsum[i]:dof_cumsum[i+1]] = self.vecX_i(i=i)
		return resX
			

	def XKY_vec(self):
		Ky = self.KY_vec()
		X = self.X_vec()
		XKY_vec = X.T.dot(Ky)
		XKY = self.XKY()
		diff = XKY_vec - XKY
		if np.absolute(diff).sum()>self.diff_threshold:
			raise Exception("missmatch, diff = %f" % np.absolute(diff).sum())
		return XKY_vec

	def KY_vec(self):
		y = self.Y_vec()
		K = self.K_vec()
		Ky = la.solve(K,y)
		return Ky

	def YKY_vec(self):
		y = self.Y_vec()
		Ky = self.KY_vec()
		yKy_vec = y.T.dot(Ky)
		yKy = self.YKY()
		diff = yKy_vec - yKy
		if np.absolute(diff).sum()>self.diff_threshold:
			raise Exception("missmatch, diff = %f" % np.absolute(diff).sum())
		return yKy_vec

	def KX_vec(self):
		X=self.X_vec()
		K = self.K_vec()
		KX = la.solve(K,X)
		return KX

	def XKX_vec(self):
		KX = self.KX_vec()
		X = self.X_vec()
		XKX_vec = X.T.dot(KX)
		XKX = self.XKX()
		diff = XKX_vec - XKX
		if np.absolute(diff).sum()>self.diff_threshold:
			raise Exception("missmatch, diff = %f" % np.absolute(diff).sum())
		return XKX_vec

	def beta_vec(self):
		XKX = self.XKX_vec()
		XKy = self.XKY_vec()
		beta_vec = la.solve(XKX,XKy)
		beta = self.beta()
		diff = beta_vec - beta
		if np.absolute(diff).sum()>self.diff_threshold:
			raise Exception("missmatch, diff = %f" % np.absolute(diff).sum())
		return beta

	def predict_vec(self):
		X = self.X_vec()
		beta = self.beta_vec()
		return X.dot(beta)

	def residual_vec(self):
		y = self.Y_vec()
		return y - self.predict_vec()

	def K_vec(self):
		#return self.h2*self.covariance_vec(i=0) + (1.0-self.h2)*self.covariance_vec(i=1)
		return self.covariance_vec(i=0) + self.covariance_vec(i=1)

	def resKres_vec(self):
		res = self.residual_vec()
		K = self.K_vec()
		Kres = la.solve(K,res)
		resKres_vec = res.T.dot(Kres)
		resKres = self.resKres()
		diff = resKres_vec - resKres
		if np.absolute(diff).sum()>self.diff_threshold:
			raise Exception("missmatch, diff = %f" % np.absolute(diff).sum())
		assert resKres_vec.shape[0]==1
		assert resKres_vec.shape[1]==1
		return resKres_vec[0,0]

	def logdet_XKX_vec(self):
		XKX = self.XKX_vec()
		sign,logdet_XKX_vec = la.slogdet(XKX)
		logdet_XKX = self.logdet_XKX()
		diff = logdet_XKX_vec - logdet_XKX
		if np.absolute(diff).sum()>self.diff_threshold:
			raise Exception("missmatch, diff = %f" % np.absolute(diff).sum())
		return logdet_XKX_vec


	def nLL_vec(self, reml=True):
		
		yKy = self.resKres_vec()

		logdet = self.logdet_K_vec()

		if reml:
			logdet += self.logdet_XKX_vec()
			dof = self.dof.sum()
			var = yKy/(self.N*self.P-dof)
			const = (self.N*self.P-dof) * np.log(2.0*sp.pi)
		else:
			var = yKy/(self.N*self.P)
			const = self.N*self.P * np.log(2.0*sp.pi)

		dataterm = yKy/var
		logl_vec =  -0.5 * (const + logdet + dataterm)

		logl = self.nLL(reml=reml)
		diff = logl_vec - logl
		if np.absolute(diff).sum()>self.diff_threshold:
			raise Exception("log likelihood missmatch, diff = %f" % np.absolute(diff).sum())
		return logl_vec

if __name__ == "__main__":

	plot = "plot" in sys.argv

	# generate data
	N=100
	S=5
	R=1000
	P=3
	var_K1=0.6
	var_K2=0.4
	h2_1=0.9
	h2_2=0.1

	h2 = 0.5

	A=None
	data = fast_any.GeneratorKron(N=N,S=S,R=R,P=P,var_K1=var_K1,var_K2=var_K2,h2_1=h2_1,h2_2=h2_2,h2=h2)

	X = [data.snps, np.ones((N,1))]
	A = [np.eye(data.P), np.eye(data.P)]
	A_any = [None, None]

	lmm1 = lmm_kron.LmmKronecker(Y=data.Y, R1=data.R1, C1=data.C1, R2=data.R2, C2=data.C2, X=X, A=A)
	lmm1_any = lmm_kron.LmmKronecker(Y=data.Y, R1=data.R1, C1=data.C1, R2=data.R2, C2=data.C2, X=X, A=A_any)
	lmm1_ = LmmKronecker_vec(Y=data.Y, R1=data.R1, C1=data.C1, R2=data.R2, C2=data.C2, X=X, A=A)
	#lmm2_ = LmmKronecker_vec(Y=data.Y, R1=data.R1, C1=data.C1, R2=data.R2, C2=data.C2, X=data.snps, A=data.A)
	XKX1 = lmm1.XKX()
	XKX1_any = lmm1_any.XKX()
	XKX1_ = lmm1_.XKX_vec()
	#XKX2_ = lmm2_.XKX_vec()
	mllogl1 = lmm1.nLL(reml=False )
	remllogl1 = lmm1.nLL(reml=True)
	mllogl1_any = lmm1_any.nLL(reml=False)
	remllogl1_any = lmm1_any.nLL(reml=True)
	mllogl1_ = lmm1_.nLL_vec(reml=False)
	remllogl1_ = lmm1_.nLL_vec(reml=True)
	#mllogl2_ = lmm2_.nLL_vec(reml=False )
	#remllogl2_ = lmm2_.nLL_vec(reml=True)
	diff_ml = np.absolute(mllogl1 - mllogl1_any)
	assert diff_ml<=1e-8, "ML log likelihood missmatch between any and normal"