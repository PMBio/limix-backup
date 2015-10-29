import sys
import numpy.linalg as la 
import scipy.stats as st
import numpy as np
import scipy as sp
import limix.core.association.kron_util as kron_util
import limix.core.association.kron_gwas as kron_gwas
import limix.core.association.kron_lmm as kron_lmm

import ipdb
import limix.core.fastany.fast_any as fast_any

class KroneckerLMM_vec(kron_gwas.KroneckerGWAS):
	def __init__(self, Y, R1, C1, R2, C2, X, A=None, h2=0.5, reml=True, diff_threshold=1e-8):
		"""
		Kronecker mixed model implementation performing maths operations expensively (for debugging purposes)

		model specification:

		vec(Y) ~ N( vec(sum_i vec(X_i * B_i * A_i ), sigma2 * [ h2 * C1 x R1 + (1-h2) * C2 x R2 ]  )

		Args:
			Y:		phenotypes [N x P] ndarray
			R1:		first row covariance matrix [N x N] ndarray
			C1:		first column covariance matrix [P x P] ndarray
			R2:		second row covariance matrix [N x N] ndarray
			C2:		second column covariance matrix [P x P] ndarray
			X:		list of row covariates matrices of length T, each [N x D_t] ndarray
			A:		list of column covariates matrices of length T, each either [dof_t x P] ndarray or None.
					None will result in an efficient implementation of any effect.
			h2:		heritability (default 0.5)
			reml:	use REML (True) or ML (False)? (default: True)
		"""
		kron_gwas.KroneckerGWAS.__init__(self, Y=Y, R1=R1, C1=C1, R2=R2, C2=C2, X=X, A=A, h2=h2, reml=reml)
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
		return self.h2*self.covariance_vec(i=0) + (1.0-self.h2)*self.covariance_vec(i=1)

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

	def resKres_snps_vec(self):
		YKY = self.YKY()
		XKY = self.XKY_snps_vec()
		beta_snps_vec = self.beta_snps_vec()
		resKres_snps_vec = YKY - (beta_snps_vec*XKY).sum()
		resKres_snps = self.resKres_snps()
		diff = resKres_snps - resKres_snps_vec
		if np.absolute(diff).sum()>self.diff_threshold:
			raise Exception("resKres_snps missmatch, diff = %f" % np.absolute(diff).sum())
		return resKres_snps_vec

	def logdet_XKX_vec(self):
		XKX = self.XKX_vec()
		sign,logdet_XKX_vec = la.slogdet(XKX)
		logdet_XKX = self.logdet_XKX()
		diff = logdet_XKX_vec - logdet_XKX
		if np.absolute(diff).sum()>self.diff_threshold:
			raise Exception("missmatch, diff = %f" % np.absolute(diff).sum())
		return logdet_XKX_vec

	def XKX_snps_vec(self):
		dof = self.dof.sum()
		dof_snps = self.dof_snps.sum()
		XKX_snps_vec = np.empty((dof+dof_snps,dof+dof_snps))
		XKX_snps_vec[:dof,:dof] = self.XKX()
		XKX_snps_vec[dof:,dof:] = self.snpsKsnps()
		XKX_snps_vec[dof:,:dof] = self.snpsKX()
		XKX_snps_vec[:dof,dof:] = self.snpsKX().T
		return XKX_snps_vec

	def XKY_snps_vec(self):
		dof = self.dof.sum()
		dof_snps = self.dof_snps.sum()
		XKY_snps_vec = np.empty((dof+dof_snps,1))
		XKY_snps_vec[:dof] = self.XKY()
		XKY_snps_vec[dof:] = self.snpsKY()
		#print XKY_snps_vec[dof:]
		#n2 = self.snpsKY()
		#self.A_snps = self.A_snps
		#self.X_snps = self.X_snps
		#diff1 = XKY_snps_vec[dof:] - self.snpsKY()
		#diff2 = XKY_snps_vec[dof:] - n2
		#print "XKY"
		#print diff1
		#print diff2
		#if (np.absolute(diff1).sum()>self.diff_threshold) or (np.absolute(diff2).sum()>self.diff_threshold):
		#	raise Exception("missmatch")
		return XKY_snps_vec

	def beta_snps_vec(self):
		XKX_snps_vec = self.XKX_snps_vec()
		XKY_snps_vec = self.XKY_snps_vec()
		beta_snps_vec = la.solve(XKX_snps_vec,XKY_snps_vec)

		beta_X = self.beta()
		beta_up, beta_snps = self.beta_snps()
		beta = np.zeros_like(beta_snps_vec)
		beta[:beta_up.shape[0]] = beta_up + beta_X
		beta[beta_up.shape[0]:] = beta_snps

		diff = beta - beta_snps_vec
		if np.absolute(diff).sum()>self.diff_threshold:
			raise Exception("beta_snps missmatch, diff = %f" % np.absolute(diff).sum())
		return beta_snps_vec

	def LL_snps_vec(self):
		yKy = self.resKres_snps_vec()
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
		logl_vec =  -0.5 * (const + logdet + dataterm)
		logl = self.LL_snps()
		diff = logl_vec - logl
		if np.absolute(diff).sum()>self.diff_threshold:
			raise Exception("log likelihood snps missmatch, diff = %f" % np.absolute(diff).sum())		
		return logl_vec

	def LL_vec(self):
		yKy = self.resKres_vec()
		logdet = self.logdet_K_vec()
		if self.reml:
			logdet += self.logdet_XKX_vec()
			dof = self.dof.sum()
			var = yKy/(self.N*self.P-dof)
			const = (self.N*self.P-dof) * (np.log(2.0*sp.pi) + np.log(var))
		else:
			var = yKy/(self.N*self.P)
			const = self.N*self.P * (np.log(2.0*sp.pi) + np.log(var))
		dataterm = yKy/var
		logl_vec =  -0.5 * (const + logdet + dataterm)
		logl = self.LL()
		diff = logl_vec - logl
		if np.absolute(diff).sum()>self.diff_threshold:
			raise Exception("log likelihood missmatch, diff = %f" % np.absolute(diff).sum())
		return logl_vec

	def lrt_snps_vec(self):
		LL_0 = self.LL()
		LL_alt = self.LL_snps_vec()
		if LL_0 > (LL_alt+1e-8):
			raise Exception("null model likelihood is smaller than alternative model likelihood. LL_0=%f, LL_alt=%f" % (LL_0, LL_alt))
		lrt_vec = 2.0 * (LL_alt - LL_0)
		dof_lrt = self.dof_snps.sum()
		p_value_vec = st.chi2.sf(lrt_vec,dof_lrt)
		lrt,p_value = self.lrt_snps()
		diff_p = np.log(p_value)-np.log(p_value_vec)
		if np.absolute(diff_p)>self.diff_threshold:
			raise Exception("p_value missmatch, diff = %f" % np.absolute(diff_p))
		diff_lrt = np.log(lrt)-np.log(lrt_vec)
		if np.absolute(diff_lrt).sum()>self.diff_threshold:
			raise Exception("lrt missmatch, diff = %f" % np.absolute(diff_lrt))
		return (lrt_vec, p_value_vec)

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

	h2_sim = 0.5

	A=None
	data = fast_any.GeneratorKron(N=N,S=S,R=R,P=P,var_K1=var_K1,var_K2=var_K2,h2_1=h2_1,h2_2=h2_2,h2=h2_sim)

	h2 = data.var_K1/(data.var_K1+data.var_K2)
	X = [data.snps[:,:-1], np.ones((N,1))]
	A = [np.eye(data.P), np.eye(data.P)]
	A_any = [None, None]

	lmm1 = kron_lmm.KroneckerLMM(Y=data.Y, R1=data.R1, C1=data.C1, R2=data.R2, C2=data.C2, X=X, A=A, h2=h2)
	lmm1_any = kron_lmm.KroneckerLMM(Y=data.Y, R1=data.R1, C1=data.C1, R2=data.R2, C2=data.C2, X=X, A=A_any, h2=h2)
	lmm1_ = KroneckerLMM_vec(Y=data.Y, R1=data.R1, C1=data.C1, R2=data.R2, C2=data.C2, X=X, A=A, h2=h2)
	#lmm2_ = LmmKronecker_vec(Y=data.Y, R1=data.R1, C1=data.C1, R2=data.R2, C2=data.C2, X=data.snps, A=data.A)
	XKX1 = lmm1.XKX()
	XKX1_any = lmm1_any.XKX()
	XKX1_ = lmm1_.XKX_vec()
	#XKX2_ = lmm2_.XKX_vec()
	lmm1.reml=False
	mllogl1 = lmm1.LL()
	lmm1.reml=True
	remllogl1 = lmm1.LL()
	lmm1_any.reml=False
	mllogl1_any = lmm1_any.LL()
	lmm1_any.reml=True
	remllogl1_any = lmm1_any.LL()
	lmm1_.reml=False
	mllogl1_ = lmm1_.LL_vec()
	lmm1_.reml=True
	remllogl1_ = lmm1_.LL_vec()

	#mllogl2_ = lmm2_.nLL_vec(reml=False )
	#remllogl2_ = lmm2_.nLL_vec(reml=True)
	diff_ml = np.absolute(mllogl1 - mllogl1_any)
	assert diff_ml<=1e-8, "ML log likelihood missmatch between any and normal"


	X_snps = [data.snps[:,-1:]]
	A_snps = [np.eye(data.P)]
	A_snps_any = [None]
	gwas = kron_gwas.KroneckerGWAS(Y=data.Y, R1=data.R1, C1=data.C1, R2=data.R2, C2=data.C2, X=X, A=A_any, X_snps=X_snps, A_snps=A_snps, h2=h2)
	gwas_vec = KroneckerLMM_vec(Y=data.Y, R1=data.R1, C1=data.C1, R2=data.R2, C2=data.C2, X=X, A=A_any, h2=h2)
	gwas_vec.A_snps = A_snps
	gwas_vec.X_snps = X_snps
	
	LL2 = gwas.LL()
	LL2_snps = gwas.LL_snps()
	lrt2,pv2 = gwas.lrt_snps()
	lrt2_vec,pv2_vec = gwas_vec.lrt_snps_vec()
	beta_up,beta_snps = gwas.beta_snps()

	gwas.A_snps = A_snps_any
	gwas_vec.A_snps = A_snps_any
	lrt2_any,pv2_any = gwas.lrt_snps()
	lrt2_any_vec,pv2_any_Vec = gwas_vec.lrt_snps_vec()
	beta_up_any,beta_snps_any = gwas.beta_snps()

	snps_0 = np.random.randn(N,10000)
	snps_test = np.concatenate((data.snps[:,-1:],snps_0),1)
	lrts,p_values = gwas.run_gwas(snps=snps_test, A_snps=None)
	
