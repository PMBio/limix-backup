import sys
import numpy.linalg as la 
import numpy as np
import scipy as sp
import scipy.stats as st
from limix.core.type.cached import cached
import limix.core.association.kron_util as kron_util
import limix.core.association.kron_lmm as kron_lmm

class KroneckerGWAS(kron_lmm.KroneckerLMM):
	def __init__(self, Y, R1, C1, R2, C2, X, A=None, var_K1=1.0, var_K2=1.0, X_snps=None, A_snps=None):
		kron_lmm.KroneckerLMM.__init__(self, Y=Y, R1=R1, C1=C1, R2=R2, C2=C2, X=X, A=A, var_K1=var_K1, var_K2=var_K2)
		
		if X_snps is not None:
			self.X_snps = X_snps
			self.A_snps = A_snps

	@property
	def dof_snps(self):
		dof = self.dof_A(A=self.A_snps) * self.dof_X(X=self.X_snps)
		return dof

	@property
	def A_snps(self):
		return self._A_snps

	@A_snps.setter
	def A_snps(self, value):
		self.clear_cache("A_snps")
		if type(value) is list:
			self._A_snps = value
		else:
			self._A_snps = [value]

	@property
	def X_snps(self):
		return self._X_snps

	@X_snps.setter
	def X_snps(self, value):
		self.clear_cache("X_snps")
		if type(value) is list:
			self._X_snps = value
		else:
			self._X_snps = [value]


	@cached(["X_snps","R"])
	def X_snps_rot(self):
		res = []
		for i in xrange(self.length_snps):
			res.append(self.rotate_X(X=self.X_snps[i]))
		return res

	@cached(["A_snps","C"])
	def A_snps_rot(self):
		res = []
		for i in xrange(self.length_snps):
			res.append(self.rotate_A(A=self.A_snps[i]))
		return res

	@property
	def length_snps(self):
		assert len(self.X_snps)==len(self.A_snps), "missmatch between len(X_snps)=%i and len(A_snps)=%i" % (len(X_snps), len(A_snps))
		return len(self.X_snps)

	#these are needed to test SNPs:
	@cached(["Y","C","R","X_snps","A_snps"])
	def snpsKY(self):
		dof = self.dof_snps
		dof_cumsum = np.concatenate(([0],dof.cumsum()))
		snpsKY = np.empty((dof.sum(),1))
		DY = self.D_rot() * self.Yrot()
		for i in xrange(self.length_snps):
			snpsKY[dof_cumsum[i]:dof_cumsum[i+1],0] = kron_util.vec(kron_util.compute_XYA(DY=DY, X=self.X_snps_rot()[i], A=self.A_snps_rot()[i]))		
		return snpsKY

	@cached(["C","R","X_snps","A_snps"])
	def snpsKsnps(self):
		dof = self.dof_snps
		dof_cumsum = np.concatenate(([0],dof.cumsum()))
		snpsKsnps = np.empty((dof.sum(),dof.sum()))
		for i in xrange(self.length_snps):
			snpsKsnps[dof_cumsum[i]:dof_cumsum[i+1],dof_cumsum[i]:dof_cumsum[i+1]] = kron_util.compute_X1KX2(Y=self.Y, D=self.D_rot(), A1=self.A_snps_rot()[i], A2=self.A_snps_rot()[i], X1=self.X_snps_rot()[i], X2=self.X_snps_rot()[i])
			for j in xrange(i+1,self.length_snps):
				snpsKsnps[dof_cumsum[i]:dof_cumsum[i+1],dof_cumsum[j]:dof_cumsum[j+1]] = kron_util.compute_X1KX2(Y=self.Y, D=self.D_rot(), A1=self.A_snps_rot()[i], A2=self.A_snps_rot()[j], X1=self.X_snps_rot()[i], X2=self.X_snps_rot()[j])
				snpsKsnps[dof_cumsum[j]:dof_cumsum[j+1],dof_cumsum[i]:dof_cumsum[i+1]] = snpsKsnps[dof_cumsum[i]:dof_cumsum[i+1],dof_cumsum[j]:dof_cumsum[j+1]].T
		return snpsKsnps

	@cached(["X","A","C","R","X_snps","A_snps"])
	def snpsKX(self):
		dof = self.dof
		dof_cumsum = np.concatenate(([0],dof.cumsum()))
		dof_snps = self.dof_snps
		dof_snps_cumsum = np.concatenate(([0],dof_snps.cumsum()))
		snpsKX = np.empty((dof_snps.sum(),dof.sum()))
		for i in xrange(self.length_snps):
			for j in xrange(self.length):
				snpsKX[dof_snps_cumsum[i]:dof_snps_cumsum[i+1],dof_cumsum[j]:dof_cumsum[j+1]] = kron_util.compute_X1KX2(Y=self.Y, D=self.D_rot(), A1=self.A_snps_rot()[i], A2=self.A_rot()[j], X1=self.X_snps_rot()[i], X2=self.X_rot()[j])
		return snpsKX

	@cached(["X","A","Y","C","R","X_snps","A_snps"])
	def beta_snps(self):
		XKX = self.XKX()
		XKY = self.XKY()
		beta = self.beta()
		snpsKsnps = self.snpsKsnps()
		snpsKX = self.snpsKX()
		snpsKY = self.snpsKY()
		AiB = la.solve(XKX,snpsKX.T)
		snpsKsnps_ = snpsKsnps - snpsKX.dot(AiB)
		snpsKY_ = snpsKY - AiB.T.dot(XKY)
		beta_snps = la.solve(snpsKsnps_,snpsKY_)
		beta_up =  -AiB.dot(beta_snps)
		#beta_X = beta + beta_up
		return (beta_up, beta_snps)

	@cached(["X","A","Y","C","R","X_snps","A_snps"])
	def resKres_snps(self):
		resKres = self.resKres()
		beta_up, beta_snps = self.beta_snps()
		XKy = self.XKY()
		snpsKy = self.snpsKY()
		rss = resKres - (beta_up*XKy).sum() - (beta_snps*snpsKy).sum()
		return rss

	@cached(["X","A","Y","C","R","X_snps","A_snps"])
	def LL_snps(self, reml=True):
		yKy = self.resKres_snps()
		logdet = self.logdet_K()
		if reml:
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

	@cached(["X","A","Y","C","R","X_snps","A_snps"])
	def lrt_snps(self, reml=True):
		LL_0 = self.LL(reml=reml)
		LL_alt = self.LL_snps(reml=reml)
		if LL_0-1e-8 > LL_alt:
			raise Exception("invalid LRT")
		lrt = 2.0 * (LL_alt - LL_0)
		dof_lrt = self.dof_snps.sum()
		p_value = st.chi2.sf(lrt,dof_lrt)
		return (lrt, p_value)

	def run_gwas(self, snps, A_snps=None, reml=True):
		X_snps_sav = self.X_snps
		A_snps_sav = self.A_snps
		lrts = np.empty((snps.shape[1]))
		p_values = np.empty((snps.shape[1]))
		self.A_snps = [A_snps]
		for s in xrange(snps.shape[1]):
			snp = snps[:,s:s+1]
			self.X_snps = [snp]
			lrts[s],p_values[s] = self.lrt_snps(reml=reml)
		self.X_snps = X_snps_sav
		self.A_snps = A_snps_sav
		return lrts, p_values
