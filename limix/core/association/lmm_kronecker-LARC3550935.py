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

    def LL_snps(self, snps, Asnps=None, inter=None, identity_trick=False):
        """
        compute log likelihood for SNPs
        """
        LL_snps = np.zeros(snps.shape[1])
        LL_new = np.zeros(snps.shape[1])
        LL_snps_0 = np.zeros(snps.shape[1])
        #index_snpterm = self.gp.mean.n_terms
        if inter is None:

            LL_snps_0[:] = self._LL_0
            for i_snp in range(snps.shape[1]):
                #correlated but not identical:
                if 0:
                    self._gp.mean.clear_cache('Fstar','Astar','Xstar','Xhat',
                         'Areml','Areml_eigh','Areml_chol','Areml_inv','beta_hat','B_hat',
                         'LRLdiag_Xhat_tens','Areml_grad',
                         'beta_grad','Xstar_beta_grad','Zstar','DLZ')
                    Areml_small1 = self._gp.mean.Areml()
                LL_new[i_snp],beta,Areml1,Areml_11 = self.LML_blockwise(snp=np.dot(self._gp.mean.Lr,snps[:,i_snp:i_snp+1]), Asnp=Asnps, identity_trick=True)
                #worse:
                #LL_new[i_snp],beta = self.LML_blockwise(snp=snps[:,i_snp:i_snp+1], Asnp=Asnps, identity_trick=True)
                if 0:
                    Areml_small2 = self._gp.mean.Areml()
                    diff_small = Areml_small1-Areml_small2
                    absdiff_small=np.absolute(diff_small)
                    print((np.max(absdiff_small)))
                self._gp.mean.addFixedEffect(F=snps[:,i_snp:i_snp+1], A=Asnps)
                
                if 0:
                    self._gp.mean.clear_cache('Fstar','Astar','Xstar','Xhat',
                         'Areml','Areml_eigh','Areml_chol','Areml_inv','beta_hat','B_hat',
                         'LRLdiag_Xhat_tens','Areml_grad',
                         'beta_grad','Xstar_beta_grad','Zstar','DLZ')
                if 0:
                    Areml2_prior = self._gp.mean.Areml()
                LL__ = self._gp.LML(identity_trick=identity_trick)
                
                Areml2 = self._gp.mean.Areml()
                diff=Areml1-Areml2
                if 0:
                    diff_prior=Areml2 - Areml2_prior
                    absdiff_prior=np.absolute(diff_prior)
                    print((np.max(absdiff_prior)))
                absdiff=np.absolute(diff)
                print((np.max(absdiff)))
                print((np.max(absdiff[0:-1,0:-1])))
                if 0:
                    diff_large_small = Areml1[0:-1,0:-1]-Areml_small2
                    absdiff_large_small=np.absolute(diff_large_small)
                    print((np.max(absdiff_large_small)))
                    diff_large_small = Areml2[0:-1,0:-1]-Areml_11
                    absdiff_large_small=np.absolute(diff_large_small)
                    print((np.max(absdiff_large_small)))
                #import ipdb;ipdb.set_trace()
                LL_snps[i_snp] = LL__
                self._gp.mean.removeFixedEffect()
        else:
            for i_snp in range(snps.shape[1]):
                self._gp.mean.addFixedEffect(F=snps[:,i_snp:i_snp+1], A=Asnps)
                LL_snps_0[i_snp] = self._gp.LML(identity_trick=identity_trick)
                self._gp.mean.addFixedEffect(F=snps[:,i_snp:i_snp+1]*inter, A=Asnps)
                LL_snps[i_snp] = self._gp.LML(identity_trick=identity_trick)
                self._gp.mean.removeFixedEffect()
                self._gp.mean.removeFixedEffect()
        return LL_snps,LL_snps_0,LL_new


    def LML_blockwise(self, snp, Asnp=None, identity_trick=False, *kw_args):
        """
        calculate LML
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
        for term in range(self._gp.mean.n_terms):
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
        AXcovarXsnp = self._gp.mean.Areml_solve(XcovarXsnp)
        XsnpXsnp_ = XsnpXsnp - XcovarXsnp.T.dot(AXcovarXsnp)
        #compute beta
       
        #compute a
        snpKY = compute_XYA(DY=self._gp.mean.Yhat(), X=snp, A=Asnp)
        if 0:
            XsnpXsnp_solver = psd_solve.psd_solver(XsnpXsnp_, lower=True, threshold=1e-10,check_finite=True,overwrite_a=False)
            #solve XsnpXsnp \ AXcovarXsnp*beta
            DCbeta = XsnpXsnp_solver.solve(XcovarXsnp.T.dot(beta),overwrite_b=True)
            #solve XsnpXsnp \ a
            Da = XsnpXsnp_solver.solve(snpKY,overwrite_b=False)
            beta_snp = Da-DCbeta
            beta_up = XcovarXsnp.dot(-beta_snp)
            var_expl_snp = (XKY*(beta + beta_up)).sum() + (snpKY*beta_snp).sum()
            beta_all = np.concatenate([beta+beta_up,beta_snp])
        
        

        Areml_11 = self._gp.mean.Areml()
        Areml_12 = XcovarXsnp
        Areml_22 = XsnpXsnp
        Areml1 = np.concatenate((Areml_11,Areml_12),1)
        Areml2 = np.concatenate((Areml_12.T,Areml_22),1)
        Areml_all = np.concatenate((Areml1,Areml2),0)
        Areml_all_solver = psd_solve.psd_solver(Areml_all, lower=True, threshold=1e-10,check_finite=True,overwrite_a=False)
        XKY_all = np.concatenate((XKY,snpKY),0)
        beta_all_ = Areml_all_solver.solve(XKY_all)
        var_expl_all=(XKY_all*beta_all_).sum()

        var_res = var_total - var_expl_all#var_expl_snp
        #import ipdb;ipdb.set_trace()
        lml += var_res
        lml *= 0.5

        #import ipdb;ipdb.set_trace()
        return lml,beta_all_,Areml_all,Areml_11

    def test_snps(self, snps, Asnps=None, inter=None, identity_trick=False):
        """
        test snps for association
        """
        if Asnps is None:
            dof = self._gp.mean.P
        else:
            dof = Asnps.shape[0]
        LL_snps, LL_snps_0,LL_new = self.LL_snps(snps=snps,Asnps=Asnps, inter=inter, identity_trick=identity_trick)
        LRT = 2.0 * (LL_snps_0 - LL_snps)
        LRT_new = 2.0 * (LL_snps_0 - LL_new)
        pv = stats.chi2.sf(LRT,dof)
        pv_new = stats.chi2.sf(LRT_new,dof)
        #import ipdb;ipdb.set_trace()
        return pv,LL_snps,LL_snps_0,LL_new,pv_new



