from covar_base import Covariance
import pdb
import numpy as np
import scipy as sp
from acombinators import ACombinatorCov
from limix.core.type.cached import Cached, cached

class CovMultiKronSum(ACombinatorCov):
    """
    Sum of multiple kronecker covariances matrices.
    (Column covariances are )
    The number of paramteters is the sum of the parameters of the single column covariances.
    """

    def __init__(self, C, R):
        """
        Args:
            C:     list of column covariances 
            R:     list of row covariances
        """
        assert len(C)==len(R), 'CovMultiKronSum: C and R must have the same length'
        self._dim_c = C[0].dim
        self._dim_r = R[0].shape[0]
        self._n_terms = len(C)
        ACombinatorCov.__init__(self)
        self.dim = self._dim_c * self._dim_r
        for term_i in range(self.n_terms):
            assert C[term_i].dim==self._dim_c, 'CovMultiKronSum:: Dimension mismatch'
            assert R[term_i].shape[0]==self._dim_r, 'CovMultiKronSum:: Dimension mismatch'
            assert R[term_i].shape[1]==self._dim_r, 'CovMultiKronSum:: Dimension mismatch'
            self.covars.append(C[term_i])
            C[term_i].register(self.clear_all)
        self._R = sp.array(R)

    #####################
    # Covars handling
    #####################
    def addCovariance(self,covar):
        raise NotImplementedError("This method is not available for CovMultiKronSum")

    #####################
    # Get row and col covars 
    #####################
    @property
    def n_terms(self):
        return self._n_terms

    @property
    def dim_r(self):
        return self._dim_r

    @property
    def dim_c(self):
        return self._dim_c

    @property
    def R(self):
        return self._R

    @property
    def C(self):
        return self.covars

    @property
    @cached('covar_base')
    def C_K(self):
        RV = sp.zeros((self.n_terms, self.dim_c, self.dim_c))
        for ti in range(self.n_terms):
            RV[ti, :, :] = self.C[i].K()
        return RV

    #####################
    # Cached
    #####################
    @cached('covar_base')
    def K(self):
        K = sp.zeros((self.dim,self.dim))
        for i in range(len(self.covars)):
            K += sp.kron(self.C[i].K(), self.R[i])
        return K

    @cached('covar_base')
    def Kcross(self):
        raise NotImplementedError("Not implemented yet")

    @cached('covar_base')
    def K_grad_i(self,i):
        istart = 0
        for j in range(len(self.covars)):
            istop = istart + self.getCovariance(j).getNumberParams()
            if (i < istop):
                idx = i - istart
                return sp.kron(self.C[j].K_grad_i(idx), self.R[j])
            istart = istop
        return None

    @cached('covar_base')
    def K_hess_i_j(self, i, j):
        istart = 0
        jstart = 0
        for c1 in range(len(self.covars)):
            istop = istart + self.getCovariance(c1).getNumberParams()
            if (i < istop):
                i0 = i - istart
                break
            istart = istop
        for c2 in range(len(self.covars)):
            jstop = jstart + self.getCovariance(c2).getNumberParams()
            if (j < jstop):
                j0 = j - jstart
                break
            jstart = jstop
        if c1==c2:
            r = sp.kron(self.C[c1].K_hess_i_j(i0, j0), self.R[c1])
        else:
            r = sp.zeros((self.dim, self.dim))
        return r

    ####################
    # Non-cached methods
    ####################
    def dot(self, M):
        pdb.set_trace()
        vei_M = M.reshape((self.dim_r, self.dim_c, M.shape[1]), order='F') 
        R_veiM = sp.tensordot(self.R, vei_M, (1, 0))
        R_veiM_C = sp.tensordot(R_veiM, self.C_K, ((0, 2), (0, 1)))
        return R_veiM_C.reshape(M.shape, order='F') 

    ####################
    # Interpretable Params
    ####################
    def K_grad_interParam_i(self,i):
        istart = 0
        for j in range(len(self.covars)):
            istop = istart + self.getCovariance(j).getNumberParams()
            if (i < istop):
                idx = i - istart
                return self.getCovariance(j).K_grad_interParam_i(idx)
            istart = istop
        return None

