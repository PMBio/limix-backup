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
        if Asnps is not None:
            if (Asnps.shape[0]==Asnps.shape[1]) and Asnps == np.eye(Asnps.shape[0]):
                pass
        LL_snps = np.zeros(snps.shape[1])
        LL_new = np.zeros(snps.shape[1])
        LL_snps_0 = np.zeros(snps.shape[1])
        identity_trick_prev = self._gp.mean.identity_trick
        self._gp.mean.use_identity_trick(identity_trick=identity_trick)
        if not identity_trick:
            if Asnps is not None:
                Asnps.dot(self._gp.mean.Lc.T)
            else:
                Asnps = self._gp.mean.Lc.T
        if inter is None:
            LL_snps_0[:] = self._LL_0
            for i_snp in range(snps.shape[1]):

                LL_snps[i_snp],beta = self.LML_blockwise(snp=np.dot(self._gp.mean.Lr,snps[:,i_snp:i_snp+1]), Asnp=Asnps)
                if False and (Asnps is None) and (self._gp.mean):
                    LL_ = self.LML_snps_blockwise_any_singlesnp(snp=np.dot(self._gp.mean.Lr,snps[:,i_snp:i_snp+1]))

        else:
            for i_snp in range(snps.shape[1]):
                self._gp.mean.addFixedEffect(F=snps[:,i_snp:i_snp+1], A=Asnps)
                LL_snps_0[i_snp] = self._gp.LML()
                self._gp.mean.addFixedEffect(F=snps[:,i_snp:i_snp+1]*inter, A=Asnps)
                LL_snps[i_snp] = self._gp.LML()
                self._gp.mean.removeFixedEffect()
                self._gp.mean.removeFixedEffect()
        self._gp.mean.use_identity_trick(identity_trick=identity_trick_prev)
        return LL_snps,LL_snps_0


    def LML_snps_blockwise_any_multisnp(self, snps ):
        """
        calculate LML
        The beta of the SNP tested is computed using blockwise matrix inversion.
        """
        self._gp._update_cache()
        
        #1. const term
        lml  = self._gp.N*self._gp.P*np.log(2.0*np.pi)

        #2. logdet term
        lml += np.sum(np.log(self._gp.cache['Sc2']))*self._gp.N + np.log(self._gp.cache['s']).sum()

        #3. quadratic term
        XKY = self._gp.mean.compute_XKY(M=self._gp.mean.Yhat())
        beta = self._gp.mean.Areml_solve(XKY)
        var_total = (self._gp.mean.Yhat()*self._gp.mean.Ystar()).sum()
        var_expl_cov = (XKY*beta).sum()
        
        
        #use blockwise matrix inversion
        #[  Areml,          XcovarXsnp
        #   XcovarXsnp.T    XsnpXsnp    ]
        var_expl_snp = np.zeros((snps.shape[1]))
        var_expl_up = 0.0

        #trivially parallelizable:
        for p in range(self._gp.P):
            Dsnps = snps * D[:,c:c+1]
            XsnpKXsnp = (Dsnps * snps).sum(0)
            XcovarXsnp = np.zeros((self._gp.mean.dof,snps.shape[1]))
            start = 0
            if 0:
                for term in range(self._gp.mean.len):
                    n_effs_term = self._gp.mean.Fstar()[term].shape[1]
            
                    Astar_term = self._gp.mean.Astar()[term]
                    n_effs_term *= Astar_term.shape[0]
                    stop = start + n_effs_term
                    block = self._gp.mean.Fstar()[term].T.dot(Dsnps)
                    block = np.kron(A2[:,c:c+1],block)
                    XcovarXsnp[start:stop,:] = block
                    start=stop
            XanyXsnp = self._gp.mean.Fstar_any.T.dot(Dsnps)

            #For any effect covariates perform low rank update for each phenotype
            
            #compute DiC:
            DiC = self._gp.mean.Areml_solver.solve(b_any=XanyXsnp,p=p)

            #compute the Schur complement:
            up_schur = (XanyXsnp * DiC).sum(0)
            schur = XsnpKXsnp - up_schur
            
            beta_snp = (XsnpKY - DiC.T.dot(XKY)) / schur
            beta_up = DiC.T.dot(beta_snp)

            var_expl_snp += (XsnpKY * beta_snp)
            var_expl_up += (XKY * beta_up).sum()

        var_res = var_total - var_expl_snp - var_expl_cov - var_expl_up
        
        lml += var_res
        lml *= 0.5
            
        return lml,beta_all

    def LML_snps_blockwise_any_singlesnp(self, snp ):
        """
        calculate LML
        The beta of the SNP tested is computed using blockwise matrix inversion.
        """
        self._gp._update_cache()
        
        #1. const term
        lml  = self._gp.N*self._gp.P*np.log(2.0*np.pi)

        #2. logdet term
        lml += np.sum(np.log(self._gp.cache['Sc2']))*self._gp.N + np.log(self._gp.cache['s']).sum()

        #3. quadratic term
        XKY = self._gp.mean.compute_XKY(M=self._gp.mean.Yhat())
        beta = self._gp.mean.Areml_solve(XKY)
        var_total = (self._gp.mean.Yhat()*self._gp.mean.Ystar()).sum()
        var_expl_cov = (XKY*beta).sum()
        
        
        #use blockwise matrix inversion
        #[  Areml,          XcovarXsnp
        #   XcovarXsnp.T    XsnpXsnp    ]
        Dsnp = snp * self._gp.mean.D
        XsnpKXsnp = (Dsnp * snp).sum(0)
        XcovarXsnp = np.zeros((self._gp.mean.dof,snps.shape[1]))
        start = 0
        if 0:
            for term in range(self._gp.mean.len):
                n_effs_term = self._gp.mean.Fstar()[term].shape[1]
            
                Astar_term = self._gp.mean.Astar()[term]
                n_effs_term *= Astar_term.shape[0]
                stop = start + n_effs_term
                block = self._gp.mean.Fstar()[term].T.dot(Dsnps)
                block = np.kron(A2[:,c:c+1],block)
                XcovarXsnp[start:stop,:] = block
                start=stop
        XanyXsnp = self._gp.mean.Fstar_any.T.dot(Dsnps)
        XKXsnp = XXX
        #For any effect covariates perform low rank update for each phenotype
            
        #compute DiC:
        DiC = self._gp.mean.Areml_solver.solve(b_any=XanyXsnp)

        #compute the Schur complement:
        up_schur = XanyXsnp.T.dot(DiC)
        schur = XsnpKXsnp - up_schur
            
        beta_snp = (XsnpKY - DiC.T.dot(XKY)) / schur
        beta_up = DiC.T.dot(beta_snp)

        var_expl_snp = (XsnpKY * beta_snp).sum()
        var_expl_up = (XKY * beta_up).sum()

        var_res = var_total - var_expl_snp - var_expl_cov - var_expl_up
        
        lml += var_res
        lml *= 0.5
            
        return lml,beta_all


    def LML_blockwise(self, snp, Asnp=None, *kw_args):
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
        
        XKY = self._gp.mean.compute_XKY(M=self._gp.mean.Yhat())
        import ipdb; ipdb.set_trace()
        beta = self._gp.mean.Areml_solve(XKY)
        var_total = (self._gp.mean.Yhat()*self._gp.mean.Ystar()).sum()
        var_expl = (XKY*beta).sum()
        
        
        #use blockwise matrix inversion
        #[  Areml,          XcovarXsnp
        #   XcovarXsnp.T    XsnpXsnp    ]
        
        XsnpXsnp = compute_X1KX2(Y=self._gp.mean.Ystar(), D=self._gp.mean.D, X1=snp, X2=snp, A1=Asnp, A2=Asnp)
        XcovarXsnp = np.zeros((self._gp.mean.n_fixed_effs,nW_Asnp*snp.shape[1]))
        start = 0
        for term in range(self._gp.mean.n_terms):
            n_effs_term = self._gp.mean.Fstar()[term].shape[1]
            
            if self._gp.mean.identity_trick and self._gp.mean.A_identity[term]:
                Astar_term = None
                n_effs_term *= self._gp.P
            else:
                Astar_term = self._gp.mean.Astar()[term]
                n_effs_term *= Astar_term.shape[0]
            stop = start + n_effs_term
            block = compute_X1KX2(Y=self._gp.mean.Ystar(), D=self._gp.mean.D, X1=self._gp.mean.Fstar()[term], X2=snp, A1=Astar_term, A2=Asnp)
            XcovarXsnp[start:stop,:] = block
            start=stop
        AXcovarXsnp = self._gp.mean.Areml_solve(XcovarXsnp)
        XsnpXsnp_ = XsnpXsnp - XcovarXsnp.T.dot(AXcovarXsnp)
        
        #compute a
        snpKY = compute_XYA(DY=self._gp.mean.Yhat(), X=snp, A=Asnp).ravel(order='F')
        

        XsnpXsnp_solver = psd_solve.psd_solver(XsnpXsnp_, lower=True, threshold=1e-10,check_finite=True,overwrite_a=False)
        #solve XsnpXsnp \ AXcovarXsnp*beta

        if 0:
            beta_snp1 = XsnpXsnp_solver.solve(snpKY,overwrite_b=False)
            beta_snp2 = XsnpXsnp_solver.solve(AXcovarXsnp.T.dot(XKY),overwrite_b=False)
            beta_snp = beta_snp1 - beta_snp2
        beta_snp = XsnpXsnp_solver.solve(snpKY - AXcovarXsnp.T.dot(XKY),overwrite_b=False)

        #solve XsnpXsnp \ a
        if 0:
            beta_up1 = AXcovarXsnp.dot(beta_snp2) #AXcovarXsnp.dot(beta_snp)#self._gp.mean.Areml_solve(XcovarXsnp.dot(-beta_snp),identity_trick=identity_trick)#This is not correct
            beta_up2 = AXcovarXsnp.dot(beta_snp1)
            beta_up = beta_up1 - beta_up2
            beta_new = beta+beta_up
        beta_up = - AXcovarXsnp.dot(beta_snp)
        var_expl_update = (XKY*beta_up).sum()
        var_expl_snp = (snpKY*beta_snp).sum()

        var_expl_all = var_expl + var_expl_snp+var_expl_update
        beta_all = np.concatenate([beta+beta_up,beta_snp])
        var_res = var_total - var_expl_snp - var_expl_update - var_expl

        lml += var_res
        lml *= 0.5
        #if (var_expl_all)<var_expl-1e-4:
        if 0: #debugging
            Areml1 = np.concatenate((self._gp.mean.Areml(),XcovarXsnp),1)
            Areml2 = np.concatenate((XcovarXsnp.T, XsnpXsnp),1)
            Areml_all = np.concatenate((Areml1,Areml2),0)
            XKY_all = np.concatenate((XKY,snpKY),0)

            Areml_all_solver = psd_solve.psd_solver(Areml_all, lower=True, threshold=1e-10,check_finite=True,overwrite_a=False)
            beta_all_ = Areml_all_solver.solve(XKY_all)
            Areml_all_inv = Areml_all_solver.solve(np.eye(XKY_all.shape[0]))
            Areml_inv = self._gp.mean.Areml_inv()
            Aremlinv_up = AXcovarXsnp.dot(XsnpXsnp_solver.solve(AXcovarXsnp.T))
            Areml_inv_part_1 = Areml_inv + Aremlinv_up
            Areml_inv_part_2 = XsnpXsnp_solver.solve(np.eye(XsnpXsnp.shape[0]))
            Areml_inv_part_3 = - XsnpXsnp_solver.solve(AXcovarXsnp.T)
            Areml_all_inv_1 = np.concatenate((Areml_inv_part_1,Areml_inv_part_3.T),1)
            Areml_all_inv_2 = np.concatenate((Areml_inv_part_3,Areml_inv_part_2),1)
            Areml_all_inv_ = np.concatenate((Areml_all_inv_1,Areml_all_inv_2),0)
            beta_all__ = np.dot(Areml_all_inv_,XKY_all)
            diff_areml =  np.absolute(Areml_all_inv_-Areml_all_inv).sum()
            diff_beta_ = np.absolute(beta_all_-beta_all).sum()
            diff_beta__ = np.absolute(beta_all__-beta_all).sum()
            diff_beta_hat = np.absolute(beta-beta___).sum()
            print(("var_expl = %.4f" % var_expl))
            print(("var_expl_update = %.4f" % var_expl_update))
            print(("var_expl_snp = %.4f",  var_expl_snp))
            print(("absdiff Areml = %.5f",diff_areml))
            print(("absdiff beta_ = %.5f",diff_beta_))
            print(("absdiff beta__ = %.5f",diff_beta__))
            print(("absdiff beta_hat = %.5f",diff_beta_hat))
            import ipdb;ipdb.set_trace()
            
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



