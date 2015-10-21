import numpy as np
import scipy as sp
import scipy.linalg as la
from limix.core.type.cached import cached
from lowrank import LowRankCov
import warnings
import pdb

class Rank1MCov(LowRankCov):
    """
    Rank 1 semi-definite positive matrix with negative correlation.
    A Rank1M matrix is a 2x2 semi-definite rank1 matrix parameters where rho=-1
    """
    def __init__(self):
        """
        Args:
            dim:        dimension of the low-rank covariance
        """
        LowRankCov.__init__(self, 2, 1)
        self._signs = sp.array([[1.], [-1.]])

    #####################
    # Properties
    #####################
    @property
    def X(self):
        return sp.exp(self.params.reshape((self.dim, self.rank), order='F')) * self._signs

    @property
    def X_ste(self):
        print 'Implement me'

    @X.setter
    def X(self, value):
        assert self.X.shape[0]==self.dim, 'Dimension mismatch.'
        assert self.X.shape[1]==self.rank, 'Dimension mismatch.'
        if (sp.sign(value)!=self._signs).any():
            warnings.warn('Changing signs in X compatibly with Rank1MCov')
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
        return sp.reshape(Xgrad, (self.dim, self.rank), order='F') * self._signs

    @cached('covar_base')
    def K_grad_i(self,i):
        if not self._X_act:
            raise ValueError("Trying to retrieve the gradient over a "
                             "parameter that is inactive.")
        R  = sp.dot(self.X, self.Xgrad(i).T)
        R += R.T
        return R

if __name__ == '__main__':

    sp.random.seed(1)
    cov = Rank1MCov()
    cov.setRandomParams()
    print cov.K()
    print cov.K_grad_i(0)

    pdb.set_trace()
    X = sp.array([[1], [1]])
    cov.X = X
    print cov.K()

