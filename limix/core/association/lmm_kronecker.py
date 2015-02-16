import numpy as np
import scipy.linalg as la
import scipy.stats as stats
from limix.core.cobj import *
from limix.core.mean.mean import compute_X1KX2, compute_XYA
import limix.utils.psd_solve as psd_solve

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

    def LL_snps(self, snps, Asnps=None, inter=None, identity_trick=True):
        """
        compute log likelihood for SNPs
        """

        #TODO: block diagonal Areml computations efficient

        LL_snps = np.zeros(snps.shape[1])
        LL_new = np.zeros(snps.shape[1])
        LL_snps_0 = np.zeros(snps.shape[1])

        if inter is None:
            LL_snps_0[:] = self._LL_0
            for i_snp in xrange(snps.shape[1]):
                #correlated but not identical:
                LL_snps[i_snp],beta = self.LML_blockwise(snp=np.dot(self._gp.mean.Lr,snps[:,i_snp:i_snp+1]), Asnp=Asnps, identity_trick=True)
        else:
            for i_snp in xrange(snps.shape[1]):
                self._gp.mean.addFixedEffect(F=snps[:,i_snp:i_snp+1], A=Asnps)
                LL_snps_0[i_snp] = self._gp.LML(identity_trick=identity_trick)
                self._gp.mean.addFixedEffect(F=snps[:,i_snp:i_snp+1]*inter, A=Asnps)
                LL_snps[i_snp] = self._gp.LML(identity_trick=identity_trick)
                self._gp.mean.removeFixedEffect()
                self._gp.mean.removeFixedEffect()
        return LL_snps,LL_snps_0


    def LML_blockwise(self, snp, Asnp=None, identity_trick=False, *kw_args):
        """
        calculate LML
        The beta of the SNP tested is computed using blockwise matrix inversion.
        """
        self._gp._update_cache()
        
        if Asnp is None:
            nW_Asnp = self._gp.mean.Ystar().shape[1]	
        else:
            nW_Asnp = Asnp.shape[1]
            	

        #1. const term
        lml  = self._gp.N*self._gp.P*np.log(2.0*np.pi)

        #2. logdet term
        lml += np.sum(np.log(self._gp.cache['Sc2']))*self._gp.N + np.log(self._gp.cache['s']).sum()

        #3. quadratic term
        #quad1 = (self._gp.mean.Zstar(identity_trick=identity_trick)*self._gp.mean.DLZ(identity_trick=identity_trick)).sum()
        
        XKY = self._gp.mean.compute_XKY(M=self._gp.mean.Yhat(), identity_trick=identity_trick)
        beta = self._gp.mean.beta_hat(identity_trick=identity_trick)
        var_total = (self._gp.mean.Yhat()*self._gp.mean.Ystar()).sum()
        var_exp = (XKY*beta).sum()
        
        
        #use blockwise matrix inversion
        #[  Areml,          XcovarXsnp
        #   XcovarXsnp.T    XsnpXsnp    ]
        
        XsnpXsnp = compute_X1KX2(Y=self._gp.mean.Ystar(), D=self._gp.mean.D, X1=snp, X2=snp, A1=Asnp, A2=Asnp)
        XcovarXsnp = np.zeros((self._gp.mean.n_fixed_effs,nW_Asnp*snp.shape[1]))
        n_effs_sum = 0
        for term in xrange(self._gp.mean.n_terms):
            n_effs_term = self._gp.mean.Fstar()[term].shape[1]
            if self._gp.mean.A_identity[term]:
                n_effs_term *= self._gp.P
            else:
                n_effs_term *= self._gp.mean.Astar()[term].shape[1]
            if identity_trick and self._gp.mean.A_identity[term]:
                Astar_term = None
            else:
                Astar_term = self._gp.mean.Astar()[term]
            XcovarXsnp[n_effs_sum:n_effs_term+n_effs_sum,:] = compute_X1KX2(Y=self._gp.mean.Ystar(), D=self._gp.mean.D, X1=self._gp.mean.Fstar()[term], X2=snp, A1=Astar_term, A2=Asnp)
            n_effs_sum+=n_effs_term
        AXcovarXsnp = self._gp.mean.Areml_solve(XcovarXsnp)
        XsnpXsnp_ = XsnpXsnp - XcovarXsnp.T.dot(AXcovarXsnp)
        
        #compute a
        snpKY = compute_XYA(DY=self._gp.mean.Yhat(), X=snp, A=Asnp)

        XsnpXsnp_solver = psd_solve.psd_solver(XsnpXsnp_, lower=True, threshold=1e-10,check_finite=True,overwrite_a=False)
        #solve XsnpXsnp \ AXcovarXsnp*beta
        DCbeta = XsnpXsnp_solver.solve(XcovarXsnp.T.dot(beta),overwrite_b=False)
        #solve XsnpXsnp \ a
        Da = XsnpXsnp_solver.solve(snpKY,overwrite_b=False)
        beta_snp = Da-DCbeta#This is correct
        beta_up = self._gp.mean.Areml_solve(XcovarXsnp.dot(-beta_snp),identity_trick=identity_trick)#This is not correct
        var_expl_snp = (XKY*(beta + beta_up)).sum() + (snpKY*beta_snp).sum()
        beta_all = np.concatenate([beta+beta_up,beta_snp])
        var_res = var_total - var_expl_snp

        lml += var_res
        lml *= 0.5

        return lml,beta_all

    def test_snps(self, snps, Asnps=None, inter=None, identity_trick=False):
        """
        test snps for association
        """
        if Asnps is None:
            dof = self._gp.mean.P
        else:
            dof = Asnps.shape[0]
        LL_snps, LL_snps_0 = self.LL_snps(snps=snps,Asnps=Asnps, inter=inter, identity_trick=identity_trick)
        LRT = 2.0 * (LL_snps_0 - LL_snps)
        pv = stats.chi2.sf(LRT,dof)
        return pv,LL_snps,LL_snps_0



