import scipy as sp
import scipy.linalg as la
from limix.core.type.cached import cached
from covar_base import Covariance
import pdb

class LowRankCov(Covariance):
    """
    lowrank covariance
    """
    def __init__(self, dim, rank = 1):
        Covariance.__init__(self)
        self.initialize(dim, rank)
        self._initParams()

    def initialize(self, dim, rank):
        self.dim = dim
        self.rank = rank
        self._calcNumberParams()
        self._use_to_predict = False

    def clear_all(self):
        self.clear_cache('Xgrad')
        Covariance.clear_all(self)

    #####################
    # Properties
    #####################
    @property
    def X(self):
        return self.params.reshape((self.dim, self.rank), order='F')

    @property
    def X_ste(self):
        print 'Implement me'

    @X.setter
    def X(self, value):
        assert X.shape[0]==self.dim, 'Dimension mismatch.'
        assert X.shape[1]==self.rank, 'Dimension mismatch.'
        self.setParams(X.reshape(X.size, order = 'F'))

    #####################
    # Params handling
    #####################
    def _calcNumberParams(self):
        self.n_params = int(self.dim * self.rank)

    def setCovariance(self, cov):
        """ makes lowrank approximation of cov """
        assert cov.shape[0]==self.dim, 'Dimension mismatch.'
        S, U = la.eigh(cov)
        U = U[:,::-1]
        S = S[::-1]
        _X = U[:, :self.rank] * S[sp.newaxis, :self.rank]
        self.X = _X

    #####################
    # Cached
    #####################
    @cached
    def K(self):
        return sp.dot(self.X,self.X.T)

    @cached
    def Xgrad(self,i):
        Xgrad = sp.zeros(self.getNumberParams())
        Xgrad[i] = 1
        return sp.reshape(Xgrad, (self.dim, self.rank), order='F')

    @cached
    def K_grad_i(self,i):
        R  = sp.dot(self.X,self.Xgrad(i).T)
        R += R.T
        return R

    ####################
    # TODO: Interpretable Params
    ####################


if __name__ == '__main__':

    sp.random.seed(1)
    n = 4
    rank = 2
    cov = LowRankCov(n, rank)
    cov.setRandomParams()
    print cov.K()
    print cov.K_grad_i(0)
