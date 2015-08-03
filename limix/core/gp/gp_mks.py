import pdb
import scipy as sp
import scipy.linalg
import copy
import sys
import time
from gp_ls import GPLS
from limix.core.type.cached import Cached, cached
from limix.core.covar import Covariance
from limix.core.covar import CovMultiKronSum
from limix.core.mean import MeanKronSum
from limix.core.covar.cov_reml import cov_reml
from limix.utils.util_functions import vec
from limix.utils.linalg import vei_CoR_veX
from limix.core.utils import assert_type
from limix.core.utils import assert_subtype

class GPMKS(GPLS):
    """
    Gaussian Process with a covariance and a mean that are sums of multiple kronecker products:
        vec(Y) ~ N( vec( \sum_i F_i B_i A_i), \sum_j C_j \kron R_j)
    Notation:
        N = number of samples
        P = number of traits
        Y = [N, P] phenotype matrix
        F_i = sample fixed effect design for term i
        A_i = trait fixed effect design for term i
        B_i = effect sizes of fixed effect term i
        C_j = column covariance matrix for term j
        R_j = column covariance matrix for noise term j 
    """

    def __init__(self, Y, C, R, F=None, A=None):
        """
        Args:
            Y:      [N, P] phenotype matrix
            C:      list of limix trait covariances. 
                    Each term must be a limix covariance with dimension P 
            C:      list of row covariance matrices. 
                    Each term must be a numpy covariane matrix with dimension P 
            F:      list of sample fixed effect designs.
                    Each term must have first dimension N
            A:      list of trait fixed effect design.
                    Each term must have second dimension P
        """
        # assert types
        assert_type(Y, sp.ndarray, 'Y')
        assert len(C)==len(R), 'Dimension mismatch'
        for i in range(len(C)):
            assert_subtype(C[i], Covariance, 'C[%d]' % i)
            assert_type(R[i], sp.ndarray, 'R[%d]' % i)

        assert F is None and A is None, 'fixed effects not supported yet'

        # build covariance matrix
        #Iok = vec(~sp.isnan(Y))[:,0]
        #if veIok.all():     Iok = None
        covar = CovMultiKronSum(C, R) 

        #TODO: should initialize the GPLS instead
        #e.g.: GPLS.__init__(self, Y, covar)
        Cached.__init__(self)
        self.covar = covar
        self.mean = MeanKronSum(Y=Y, F=F, A=A)
        self.Areml = cov_reml(self)
        self._observe()


    def _observe(self):
        #TODO: check if we need row and col covariances
        self.covar.register(self.row_cov_has_changed, 'row_cov')
        self.covar.register(self.col_cov_has_changed, 'col_cov')
        self.mean.register(self.pheno_has_changed, 'pheno')
        #self.mean.register(self.designs_have_changed, 'designs')

    def row_cov_has_changed(self): 
        self.clear_cache('row_cov')
        self.clear_all()

    def col_cov_has_changed(self): 
        self.clear_cache('col_cov')
        self.clear_all()

    def pheno_has_changed(self): 
        self.clear_cache('pheno')
        self.clear_all()

    #######################
    # LML terms
    #######################
    def update_b(self):
        if self.mean.n_covs > 0:
            self.mean.b = self.Areml.solve(self.yKiW().T)

    @cached(['pheno', 'col_cov', 'row_cov'])
    def vei_Kiy(self):
        return self.covar.solve_ls_NxPxS(self.mean.Y)

    @cached(['pheno', 'col_cov', 'row_cov'])
    def yKiy(self):
        return float(sp.tensordot(self.mean.Y, self.vei_Kiy(), ((0,1), (0,1))))

    #@cached('gp_base')
    #def KiW(self):
    #    return self.covar.solve(self.mean.W)

    #@cached('gp_base')
    #def yKiW(self):
    #    return sp.dot(self.mean.y.T, self.KiW())

    #@cached('gp_base')
    #def KiWb(self):
    #    return sp.dot(self.KiW(), self.mean.b)

    #@cached('gp_base')
    #def yKiWb(self):
    #    return (self.mean.y*self.KiWb()).sum()

    #######################
    # gradients
    #######################
    #@cached('gp_base')
    #def DiKKiy(self, i):
    #    return sp.dot(self.covar.K_grad_i(i), self.Kiy())
    @cached(['pheno', 'col_cov', 'row_cov'])
    def R_veiKiY(self):
        RV = []
        for ti in range(self.covar.n_terms):
            RV.append(vei_CoR_veX(self.vei_Kiy()[:,:,sp.newaxis], R=self.covar.R[ti]))
        return RV

    @cached(['pheno', 'col_cov', 'row_cov'])
    def vei_DKKiy(self):
        n_params = self.getParams()['covar'].shape[0]
        RV = sp.zeros((self.covar.dim_r, self.covar.dim_c, n_params))
        pi = 0
        for ti in range(self.covar.n_terms):
            for j in range(self.covar.C[ti].getNumberParams()):
                RV[:, :, pi] = vei_CoR_veX(self.R_veiKiY()[ti], C=self.covar.C[ti].K_grad_i(j))[:,:,0]
                pi+=1
        return RV

    @cached(['pheno', 'col_cov', 'row_cov'])
    def vei_DDKKiy(self):
        n_params = self.getParams()['covar'].shape[0]
        R = sp.zeros((self.covar.dim_r, self.covar.dim_c, n_params, n_params))
        pi0 = 0
        for ti in range(self.covar.n_terms):
            pj0 = 0
            for tj in range(self.covar.n_terms):
                if ti==tj:
                    for i in range(self.covar.C[ti].getNumberParams()):
                        pi = pi0 + i
                        R[:, :, pi, pi] = vei_CoR_veX(self.R_veiKiY()[ti], C=self.covar.C[ti].K_hess_i_j(i, i))[:,:,0]
                        for j in range(i):
                            pj = pj0 + j
                            R[:, :, pi, pj] = vei_CoR_veX(self.R_veiKiY()[ti], C=self.covar.C[ti].K_hess_i_j(i, j))[:,:,0]
                            R[:, :, pj, pi] = R[:, :, pi, pj]
                pj0 += self.covar.C[tj].getNumberParams()
            pi0 += self.covar.C[ti].getNumberParams()
        return R

    def Kiy(self):
        return vec(self.vei_Kiy())

    def DKKiy(self):
        R = self.vei_DKKiy()
        return R.reshape((R.shape[0] * R.shape[1], R.shape[2]), order='F')

    def DDKKiy(self):
        R = self.vei_DDKKiy()
        return R.reshape((R.shape[0] * R.shape[1], R.shape[2], R.shape[3]), order='F')


    #@cached('gp_base')
    #def yKiy_grad_i(self, i):
    #    return -(self.Kiy()*self.DiKKiy(i)).sum()

    #@cached('gp_base')
    #def DiKKiW(self, i):
    #    return sp.dot(self.covar.K_grad_i(i), self.KiW())
    #    pass

    #@cached('gp_base')
    #def DiKKiWb(self, i):
    #    return sp.dot(self.DiKKiW(i), self.mean.b)
 
    #@cached('gp_base')
    #def yKiWb_grad_i(self, i):
    #    rv = -2*(self.Kiy()*self.DiKKiWb(i)).sum()
    #    rv += (self.KiWb()*self.DiKKiWb(i)).sum()
    #    return rv

    #######################
    # LML and gradients
    #######################
    @cached('gp_base')
    def yKiy_grad(self):
        return -sp.tensordot(self.vei_Kiy(), self.vei_DKKiy(), ((0, 1), (0, 1)))

    #@cached('gp_base')
    #def yKiWb_grad(self):
    #    n_params = self.getParams()['covar'].shape[0]
    #    RV = {'covar': sp.zeros(n_params)}
    #    if self.mean.n_covs > 0:
    #        for i in range(n_params):
    #            RV['covar'][i] = self.yKiWb_grad_i(i)
    #    return RV

    def predict(self):
        #TODO: check
        R = None
        if self.covar.use_to_predict:
            Kcross = self.covar.Kcross()
            Kiyres = self.Kiy()-self.KiWb()
            R = sp.dot(Kcross, Kiyres)
        if self.mean.use_to_predict:
            _ = self.mean.predict()
            if R is None:
                R = _
            else:
                assert _.shape[0] == R.shape[0], 'Dimension mismatch'
                assert _.shape[1] == R.shape[1], 'Dimension mismatch'
                R += _
        return R

    ############################
    # Optimize
    ############################
    def optimize(self, calc_ste=False, verbose=True, **kw_args):
        if 'tr' not in kw_args.keys():  kw_args['tr'] = 0.1
        return GPLS.optimize(self,calc_ste=calc_ste, verbose=verbose, **kw_args)

