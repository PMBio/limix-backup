import sys
import numpy.linalg as la 
import numpy as np
import scipy as sp
import scipy.stats as st
from limix.core.type.cached import cached
import limix.core.association.kron_util as kron_util
import limix.core.association.kron_lmm as kron_lmm

class KroneckerGWAS(kron_lmm.KroneckerLMM):
	def __init__(self, Y, R1, C1, R2, C2, X, A=None, h2=0.5, reml=True):
		"""
		Kronecker mixed model implementation for performing GWAS testing.
		The maths for the null model can be found in kron_lmm.KroneckerLMM

		model specification:

		vec(Y) ~ N( vec(sum_i vec(X_i * B_i * A_i ) + vec(X_snp * B_snp * A_snp), sigma2 * [ h2 * C1 x R1 + (1-h2) * C2 x R2 ]  )

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
		kron_lmm.KroneckerLMM.__init__(self, Y=Y, R1=R1, C1=C1, R2=R2, C2=C2, X=X, A=A, h2=h2, reml=reml)

	@property
	def dof_snps(self):
		"""
		list of degrees of freedom per SNP fixed effects term
		"""
		dof = self.dof_A(A=self.A_snps) * self.dof_X(X=self.X_snps)
		return dof

	@property
	def A_snps(self):
		"""
		list of rotated SNP column fixed effects design matrices A_snps 
		"""
		return self._A_snps

	@A_snps.setter
	def A_snps(self, value):
		"""
		list of rotated SNP column fixed effects design matrices A_snps 
		"""
		self.clear_cache("A_snps")
		if type(value) is list:
			self._A_snps = value
		else:
			self._A_snps = [value]

	@property
	def X_snps(self):
		"""
		list of SNP row fixed effects design matrices X_snps 
		"""
		return self._X_snps

	@X_snps.setter
	def X_snps(self, value):
		"""
		list of SNP row fixed effects design matrices X_snps 
		"""	
		self.clear_cache("X_snps")
		if type(value) is list:
			self._X_snps = value
		else:
			self._X_snps = [value]


	@cached(["X_snps","R"])
	def X_snps_rot(self):
		"""
		list of rotated SNP row fixed effects design matrices X_snps 
		"""
		res = []
		for i in xrange(self.length_snps):
			res.append(self.rotate_X(X=self.X_snps[i]))
		return res

	@cached(["A_snps","C"])
	def A_snps_rot(self):
		"""
		list of rotated SNP column fixed effects design matrices A_snps 
		"""
		res = []
		for i in xrange(self.length_snps):
			res.append(self.rotate_A(A=self.A_snps[i]))
		return res

	@property
	def length_snps(self):
		"""
		number of SNP fixed effects terms
		"""
		assert len(self.X_snps)==len(self.A_snps), "missmatch between len(X_snps)=%i and len(A_snps)=%i" % (len(X_snps), len(A_snps))
		return len(self.X_snps)

	#these are needed to test SNPs:
	@cached(["Y","C","R","X_snps","A_snps","h2"])
	def snpsKY(self):
		"""
		Returns:
			the cross product between the snp fixed effects and the phenotype Y
		"""
		dof = self.dof_snps
		dof_cumsum = np.concatenate(([0],dof.cumsum()))
		snpsKY = np.empty((dof.sum(),1))
		DY = self.D_rot() * self.Yrot()
		for i in xrange(self.length_snps):
			snpsKY[dof_cumsum[i]:dof_cumsum[i+1],0] = kron_util.vec(kron_util.compute_XYA(DY=DY, X=self.X_snps_rot()[i], A=self.A_snps_rot()[i]))		
		return snpsKY

	@cached(["C","R","X_snps","A_snps","h2"])
	def snpsKsnps(self):
		"""
		The Hessian matrix of the snps fixed effect weights beta_snps
		
		Returns:
			the cross product of the snps fixed effects with themselves
		"""
		dof = self.dof_snps
		dof_cumsum = np.concatenate(([0],dof.cumsum()))
		snpsKsnps = np.empty((dof.sum(),dof.sum()))
		for i in xrange(self.length_snps):
			snpsKsnps[dof_cumsum[i]:dof_cumsum[i+1],dof_cumsum[i]:dof_cumsum[i+1]] = kron_util.compute_X1KX2(Y=self.Y, D=self.D_rot(), A1=self.A_snps_rot()[i], A2=self.A_snps_rot()[i], X1=self.X_snps_rot()[i], X2=self.X_snps_rot()[i])
			for j in xrange(i+1,self.length_snps):
				snpsKsnps[dof_cumsum[i]:dof_cumsum[i+1],dof_cumsum[j]:dof_cumsum[j+1]] = kron_util.compute_X1KX2(Y=self.Y, D=self.D_rot(), A1=self.A_snps_rot()[i], A2=self.A_snps_rot()[j], X1=self.X_snps_rot()[i], X2=self.X_snps_rot()[j])
				snpsKsnps[dof_cumsum[j]:dof_cumsum[j+1],dof_cumsum[i]:dof_cumsum[i+1]] = snpsKsnps[dof_cumsum[i]:dof_cumsum[i+1],dof_cumsum[j]:dof_cumsum[j+1]].T
		return snpsKsnps

	@cached(["X","A","C","R","X_snps","A_snps","h2"])
	def snpsKX(self):
		"""
		Returns:
			the cross product between the snp fixed effects and the covariates fixed effects
		"""
		dof = self.dof
		dof_cumsum = np.concatenate(([0],dof.cumsum()))
		dof_snps = self.dof_snps
		dof_snps_cumsum = np.concatenate(([0],dof_snps.cumsum()))
		snpsKX = np.empty((dof_snps.sum(),dof.sum()))
		for i in xrange(self.length_snps):
			for j in xrange(self.length):
				snpsKX[dof_snps_cumsum[i]:dof_snps_cumsum[i+1],dof_cumsum[j]:dof_cumsum[j+1]] = kron_util.compute_X1KX2(Y=self.Y, D=self.D_rot(), A1=self.A_snps_rot()[i], A2=self.A_rot()[j], X1=self.X_snps_rot()[i], X2=self.X_rot()[j])
		return snpsKX

	@cached(["X","A","Y","C","R","X_snps","A_snps","h2"])
	def beta_snps(self):
		"""
		regression weight change when including the SNP in the model

		The full beta of the alternative model would correspond to 
		np.concatenate((beta + beta_up, beta_snps), 0)

		Returns:
			beta_up:	The change in regression weights for the covariates
						fixed effects when the SNP is added to the model
						1-dim ndarray of length self.dof
			beta_snps:	The regression weight for the SNP fixed effect term
						1-dim ndarray of length X_snps.shape[1] * A_snps.shape[0]
		"""
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

	@cached(["X","A","Y","C","R","X_snps","A_snps","h2"])
	def resKres_snps(self):
		"""
		residual sum of squares for the alternative model including the SNP
		"""
		resKres = self.resKres()
		beta_up, beta_snps = self.beta_snps()
		XKy = self.XKY()
		snpsKy = self.snpsKY()
		rss = resKres - (beta_up*XKy).sum() - (beta_snps*snpsKy).sum()
		return rss

	@cached(["X","A","Y","C","R","X_snps","A_snps","h2","reml"])
	def LL_snps(self):
		"""
		computes the alternative model log likelihood including the SNP using efficient low rank updates on the fixed effects.
		"""
		yKy = self.resKres_snps()
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

	@cached(["X","A","Y","C","R","X_snps","A_snps","h2","reml"])
	def lrt_snps(self):
		"""
		compute a single likelihood ratio and a P-value for

		"""
		LL_0 = self.LL()
		LL_alt = self.LL_snps()
		if LL_0-1e-8 > LL_alt:
			raise Exception("invalid LRT")
		lrt = 2.0 * (LL_alt - LL_0)
		dof_lrt = self.dof_snps.sum()
		p_value = st.chi2.sf(lrt,dof_lrt)
		return (lrt, p_value)

	def run_gwas(self, snps, A_snps=None):
		"""
		run single SNP GWAS on each of the S columns in snps.
		performing likelihood ratio tests for the null hypothesis
		H_0: X_snps * B_snp * A_snps = 0, where X_snps corresponds to a single SNP
		The test has 1*D_p degrees of freedom.

		Args:
			snps:	[N x S] ndarray of SNPs to be tested
			A_snps:	[D_p x P] column design matrix
					if set to None, an efficient any effect test is performed
					(equivalent to A_snps=np.eye(P) (default: None)

		Returns:
			lrts:		1-dim ndarray of length S of LRT statistics
			p_values:	1-dim ndarray of length S of P-values
						(comparing LRT statistic to Chi^2_D_p distribution)
		"""
		lrts = np.empty((snps.shape[1]))
		p_values = np.empty((snps.shape[1]))
		self.A_snps = [A_snps]
		for s in xrange(snps.shape[1]):
			snp = snps[:,s:s+1]
			self.X_snps = [snp]
			lrts[s],p_values[s] = self.lrt_snps()
		return lrts, p_values
