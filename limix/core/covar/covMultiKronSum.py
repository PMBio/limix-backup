from covar_base import Covariance
import pdb
import numpy as np
import scipy as sp
from acombinators import ACombinatorCov
from limix.core.type.cached import Cached, cached
import scipy.sparse.linalg as sla
from limix.utils.linalg import vei_CoR_veX

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
        self._R = R
        for term_i in range(self.n_terms):
            assert C[term_i].dim==self._dim_c, 'CovMultiKronSum:: Dimension mismatch'
            assert R[term_i].shape[0]==self._dim_r, 'CovMultiKronSum:: Dimension mismatch'
            assert R[term_i].shape[1]==self._dim_r, 'CovMultiKronSum:: Dimension mismatch'
            self.covars.append(C[term_i])
            C[term_i].register(self.col_cov_has_changed)

    def col_cov_has_changed(self):
        self.clear_all()
        self.clear_cache('col_cov')
        self._notify('col_cov')

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
    @cached('col_cov')
    def C_K(self):
        RV = sp.zeros((self.n_terms, self.dim_c, self.dim_c))
        for ti in range(self.n_terms):
            RV[ti, :, :] = self.C[ti].K()
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
        vei_M = M.reshape((self.dim_r, self.dim_c, M.shape[1]), order='F')
        return self.dot_NxPxS(vei_M).reshape(M.shape, order='F')

    def dot_NxPxS(self, M):
        """ M is NxPxS """
        RV = sp.zeros_like(M)
        for ti in range(self.n_terms):
            RV += vei_CoR_veX(M, C=self.C[ti].K(), R=self.R[ti]) 
        return RV

    def solve_ls_NxPxS(self, M, X0=None):
        # X is NxPxS tensor
        if len(M.shape)==2:     Mt = M[:, :, sp.newaxis]
        else:                   Mt = M
        if X0 is None:          X0 = 1E-3 * sp.randn(*M.shape)
        if len(X0.shape)==2:    Xt0 = X0[:, :, sp.newaxis]
        else:                   Xt0 = X0
        def veKvei(x):
            _Xt = x.reshape((self.dim_r, self.dim_c, Mt.shape[2]), order='F')
            return self.dot_NxPxS(_Xt).reshape(_Xt.size, order='F')
        Kx_O = sla.LinearOperator((Mt.size, Mt.size), matvec=veKvei, rmatvec=veKvei, dtype='float64')
        # vectorize
        m  = Mt.reshape(Mt.size, order='F')
        x0 = Xt0.reshape(Xt0.size, order='F')
        r, _ = sla.cgs(Kx_O, m, x0=x0, tol=self._tol)
        return r.reshape(M.shape, order='F')

    #####################
    # Monte Carlo methods
    #####################
    @cached('Z')
    def Z(self):
        r = sp.randn(self.dim_r, self.dim_c, self._nIterMC)
        # norm Z to improve convergence
        norm = sp.sqrt(self.dim / (float(self._nIterMC) * (r**2).sum((0,1))))
        return norm * r

    @cached(['Z', 'row_cov'])
    def RZ(self):
        RV = []
        for ti in range(self.n_terms):
            RV.append(vei_CoR_veX(self.Z(), R=self.R[ti]))
        return RV

    @cached(['row_cov', 'col_cov', 'Z'])
    def DKZ(self):
        R = sp.zeros((self.dim_r, self.dim_c, self._nIterMC, self.getNumberParams()))
        pi = 0
        for ti in range(self.n_terms):
            for j in range(self.C[ti].getNumberParams()): 
                R[:, :, :, pi] = vei_CoR_veX(self.RZ()[ti], C=self.C[ti].K_grad_i(j))
                pi+=1
        return R

    @cached(['row_cov', 'col_cov', 'Z'])
    def DDKZ(self):
        R = sp.zeros((self.dim_r, self.dim_c, self._nIterMC, self.getNumberParams(), self.getNumberParams()))
        pi0 = 0
        for ti in range(self.n_terms):
            pj0 = 0
            for tj in range(self.n_terms):
                if ti==tj: 
                    for i in range(self.C[ti].getNumberParams()): 
                        pi = pi0 + i
                        R[:, :, :, pi, pi] = vei_CoR_veX(self.RZ()[ti], C=self.C[ti].K_hess_i_j(i, i)) 
                        for j in range(i): 
                            pj = pj0 + j
                            R[:, :, :, pi, pj] = vei_CoR_veX(self.RZ()[ti], C=self.C[ti].K_hess_i_j(i, j)) 
                            R[:, :, :, pj, pi] = R[:, :, :, pi, pj] 
                pj0 += self.C[tj].getNumberParams()
            pi0 += self.C[ti].getNumberParams()
        return R

    @cached(['row_cov', 'col_cov', 'Z'])
    def KiZ(self):
        R = self.solve_ls_NxPxS(self.Z(), X0=self._KiZo)
        if self._reuse:     self._KiZo = R
        return R

    @cached(['row_cov', 'col_cov', 'Z'])
    def sample_logdet_grad(self):
        return sp.tensordot(self.DKZ(), self.KiZ(), ((0, 1, 2), (0, 1, 2)))

    @cached(['row_cov', 'col_cov', 'Z'])
    def sample_trKiDDK(self):
        return sp.tensordot(self.DDKZ(), self.KiZ(), ((0, 1, 2), (0, 1, 2)))

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

