import numpy as np
import scipy as sp
import scipy.linalg as la
from limix.core.type.cached import cached
from lowrank import LowRankCov
import warnings
import pdb

class Rank1PCov(LowRankCov):
    """
    Rank 1 semi-definite positive matrix with positive correlation.
    A Rank1P semi-definite matrix has dim * rank parameters where:
        dim = dimension of the low-rank covariance
    """
    def __init__(self):
        """
        Args:
            dim:        dimension of the low-rank covariance
        """
        LowRankCov.__init__(self, 2, 1)

    #####################
    # Properties
    #####################
    @property
    def X(self):
        return sp.exp(self.params.reshape((self.dim, self.rank), order='F'))

    @property
    def X_ste(self):
        print 'Implement me'

    @X.setter
    def X(self, value):
        assert self.X.shape[0]==self.dim, 'Dimension mismatch.'
        assert self.X.shape[1]==self.rank, 'Dimension mismatch.'
        if (value<0).any():
            warnings.warn('Setting negative entries of X to abs(X)')
        self.setParams(sp.log(abs(value).reshape(value.size, order = 'F')))

    #####################
    # Cached
    #####################
    @cached('covar_base')
    def K(self):
        return sp.dot(self.X,self.X.T)

    @cached('covar_base')
    def Xgrad(self,i):
        Xgrad = sp.zeros(self.getNumberParams())
        Xgrad[i] = sp.exp(self.getParams()[i])
        return sp.reshape(Xgrad, (self.dim, self.rank), order='F')

    @cached('covar_base')
    def K_grad_i(self,i):
        if not self._X_act:
            raise ValueError("Trying to retrieve the gradient over a "
                             "parameter that is inactive.")
        R  = sp.dot(self.X, self.Xgrad(i).T)
        R += R.T
        return R

    ####################
    # TODO: Interpretable Params
    ####################


if __name__ == '__main__':

    sp.random.seed(1)
    cov = Rank1PCov()
    cov.setRandomParams()
    print cov.K()
    print cov.K_grad_i(0)

    pdb.set_trace()
    X = sp.array([[1], [-1]])
    cov.X = X
    print cov.K()

