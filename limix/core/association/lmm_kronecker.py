import numpy as np
import scipy.linalg as la
import scipy.stats as stats
from limix.core.cobj import *

class LmmKronecker(cObject):
    pass

    def __init__(self, gp=None):
        '''
        Input:
        forcefullrank   : if True, then the code always computes K and runs cubically
					        (False)
        '''
        self._gp = gp
        
        gp.set_reml(False)#currently only ML is supported
        self._LL_0 = self._gp.LML()
        #self.clear_cache('LL_snps','test_snps')
    
    def addFixedEffect(self,F,A=None):
        self._gp.mean.addFixedEffect(F=F, A=A)
        self._LL_0 = self._gp.LML()
        #self.clear_cache('LL_snps','test_snps')
        pass

    def LL_snps(self, snps, Asnps=None, inter=None):
        """
        compute log likelihood for SNPs
        """
        LL_snps = np.zeros(snps.shape[1])
        LL_snps_0 = np.zeros(snps.shape[1])
        #index_snpterm = self.gp.mean.n_terms
        if inter is None:

            LL_snps_0[:] = self._LL_0
            for i_snp in xrange(snps.shape[1]):
                self._gp.mean.addFixedEffect(F=snps[:,i_snp:i_snp+1], A=Asnps)
                LL_snps[i_snp] = self._gp.LML()
                self._gp.mean.removeFixedEffect()
        else:
            for i_snp in xrange(snps.shape[1]):
                self._gp.mean.addFixedEffect(F=snps[:,i_snp:i_snp+1], A=Asnps)
                LL_snps_0[i_snp] = self._gp.LML()
                self._gp.mean.addFixedEffect(F=snps[:,i_snp:i_snp+1]*inter, A=Asnps)
                LL_snps[i_snp] = self._gp.LML()
                self._gp.mean.removeFixedEffect()
                self._gp.mean.removeFixedEffect()
        return LL_snps,LL_snps_0
    
    
    def test_snps(self, snps, Asnps=None):
        """
        test snps for association
        """
        if Asnps is None:
            dof = self._gp.mean.P
        else:
            dof = Asnps.shape[0]
        LL_snps, LL_snps_0 = self.LL_snps(snps=snps,Asnps=Asnps)
        LRT = 2.0 * (LL_snps_0 - LL_snps)
        pv = stats.chi2.sf(LRT,dof)
        return pv,LL_snps,LL_snps_0


def compute_D(S_C, S_R, delta=1.0):
	return 1.0 / (delta + np.outer(S_C, S_R))

def ldet_Kron(S_C, S_R, D=None, delta=1.0):
	"""
	compute the log determinant
	"""
	if D is None:
		D = compute_D(S_C=S_C, S_R=S_R, delta=delta)
	ldet = R * np.log(S_R).sum() + C * np.log(S_C).sum() - np.log(D).sum()
	return ldet


def compute_Kronecker_beta(Y, D, X, A):

    cov_beta = mean.compute_XKX()

    inv_cov_beta =la.inv(cov_beta)
    betas = np.dot(inv_cov_beta,XYA.flatten())
    #chol_beta = la.cho_factor(a=cov_beta, lower=False, overwrite_A=True, check_finite=True)
    #betas = la.cho_solve(chol_beta,b=XYA.flatten(), overwrite_b=False, check_finite=True)

    total_sos = (Y * DY).sum()
    var_expl = (betas * XYA.flatten()).sum()

    sigma2 = (res - var_expl)/(R*C) #change according to REML
    ldet = ldet_Kron(S_C, S_R, D=D, delta=1.0)

    nLL = 0.5 * ( R * C * ( np.log(2.0*np.pi) + log(sigma2) + 1.0) + ldet)

    W=[]
    cum_sum = 0
    for term in xrange(n_terms):
        current_size = X[term].shape[1] * A[term].shape[0]
        W_term = np.reshape(betas[cum_sum:cum_sum+current_size], (X[term].shape[1],A[term].shape[0]), order='C')
        cum_sum += current_size
