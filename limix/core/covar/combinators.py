from covar_base import Covariance
import pdb
import numpy as np
import scipy as SP
from acombinators import ACombinatorCov
from limix.core.type.cached import Cached, cached

class SumCov(ACombinatorCov):
    """
    Sum of multiple covariance matrices.
    The number of paramteters is the sum of the parameters of the single covariances.
    """

    def __init__(self,*covars):
        """
        Args:
            covars:     covariances to be considered in the sum
        """
        ACombinatorCov.__init__(self)
        self.dim = None
        self.covars = []
        for covar in covars:
            self.addCovariance(covar)
            covar.register(self.clear_all)

    #####################
    # Cached
    #####################
    @cached('covar_base')
    def K(self):
        K = SP.zeros((self.dim,self.dim))
        for i in range(len(self.covars)):
            K += self.getCovariance(i).K()
        return K

    @cached('covar_base')
    def Kcross(self):
        R = None
        for i in range(len(self.covars)):
            if not self.getCovariance(i).use_to_predict:    continue
            if R is None:
                R = self.covars[i].Kcross()
            else:
                _ = self.covars[i].Kcross()
                assert _.shape[0]==R.shape[0], 'Dimension mismatch.'
                assert _.shape[1]==R.shape[1], 'Dimension mismatch.'
                R += _
        return R

    @cached('covar_base')
    def K_grad_i(self,i):
        istart = 0
        for j in range(len(self.covars)):
            istop = istart + self.getCovariance(j).getNumberParams()
            if (i < istop):
                idx = i - istart
                return self.getCovariance(j).K_grad_i(idx)
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
            r = self.getCovariance(c1).K_hess_i_j(i0, j0)
        else:
            r = SP.zeros((self.dim, self.dim))
        return r

    def _calcNumberParams(self):
        self.n_params = 0
        for i in range(len(self.covars)):
            self.n_params += self.getCovariance(i).getNumberParams()
        return self.n_params

    ####################
    # Non-cached methods
    ####################
    def dot(self, M):
        R = sp.zeros_like(M)
        for i in range(len(self.covars)):
            R = self.getCovariance(i).dot(R)
        return sp.dot(self.K(), M)

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

