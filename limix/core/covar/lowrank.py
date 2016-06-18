import numpy as np
import scipy as sp
import scipy.linalg as la
from hcache import cached
from .covar_base import Covariance
import pdb

class LowRankCov(Covariance):
    """
    Low-rank semi-definite positive matrix.
    A low-rank semi-definite matrix has dim * rank parameters where:
        dim = dimension of the low-rank covariance
        rank = rank of the low-rank covariance
    """
    def __init__(self, dim, rank = 1):
        """
        Args:
            dim:        dimension of the low-rank covariance
            rank:       rank of the low-rank covariance
        """
        Covariance.__init__(self)
        self.initialize(dim, rank)

    def initialize(self, dim, rank):
        self._X_act = True
        self.dim = dim
        self.rank = rank
        self._calcNumberParams()
        self.params = np.zeros(self.n_params)
        self._use_to_predict = False

    #####################
    # Properties
    #####################
    @property
    def X(self):
        return self.params.reshape((self.dim, self.rank), order='F')

    @property
    def X_ste(self):
        print('Implement me')

    @X.setter
    def X(self, value):
        assert self.X.shape[0]==self.dim, 'Dimension mismatch.'
        assert self.X.shape[1]==self.rank, 'Dimension mismatch.'
        self.setParams(value.reshape(value.size, order = 'F'))

    #####################
    # Activation handling
    #####################
    @property
    def act_X(self):
        return self._X_act

    @act_X.setter
    def act_X(self, act):
        self._X_act = bool(act)
        self._notify()

    #####################
    # Params handling
    #####################
    def setParams(self, params):
        if not self._X_act and len(params) > 0:
            raise ValueError("Trying to set a parameter via setParams that "
                             "is not active.")
        if self._X_act:
            self.params[:] = params
            self.clear_all()

    def getParams(self):
        if not self._X_act:
            return np.array([])
        return self.params

    def getNumberParams(self):
        return int(self._X_act) * self.n_params

    def _calcNumberParams(self):
        self.n_params = int(self.dim * self.rank)

    def setCovariance(self, cov):
        """ makes lowrank approximation of cov """
        assert cov.shape[0]==self.dim, 'Dimension mismatch.'
        S, U = la.eigh(cov)
        U = U[:,::-1]
        S = S[::-1]
        _X = U[:, :self.rank] * sp.sqrt(S[:self.rank])
        self.X = _X

    #####################
    # Cached
    #####################
    @cached('covar_base')
    def K(self):
        return sp.dot(self.X,self.X.T)

    @cached('covar_base')
    def Xgrad(self,i):
        Xgrad = sp.zeros(self.getNumberParams())
        Xgrad[i] = 1
        return sp.reshape(Xgrad, (self.dim, self.rank), order='F')

    @cached('covar_base')
    def K_grad_i(self,i):
        if not self._X_act:
            raise ValueError("Trying to retrieve the gradient over a "
                             "parameter that is inactive.")

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
    print((cov.K()))
    print((cov.K_grad_i(0)))
