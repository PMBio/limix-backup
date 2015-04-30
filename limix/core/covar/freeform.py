import numpy as np
import scipy as SP
import scipy.linalg as LA
from covar_base import Covariance
from limix.core.type.cached import cached
import pdb

class FreeFormCov(Covariance):
    """
    freeform covariance function
    """
    def __init__(self,dim,jitter=1e-4):
        Covariance.__init__(self, dim)
        self.n_params = int(dim * (dim + 1.) / 2.)
        self.dim = dim
        self.params = SP.zeros(self.n_params)

        self.L = SP.zeros((self.dim,self.dim))
        self.Lgrad = SP.zeros((self.dim,self.dim))
        self.zeros = SP.zeros(self.n_params)
        self.set_jitter(jitter)

    #####################
    # Properties
    #####################
    @property
    def scale(self):
        return sp.exp(self.params[0])

    #####################
    # Params handling
    #####################
    def _calcNumberParams(self):
        self.n_params = int(0.5*self.dim*(self.dim+1))

    def set_jitter(self,value):
        self.jitter = value

    def setCovariance(self,cov):
        """ set hyperparameters from given covariance """
        chol = LA.cholesky(cov,lower=True)
        params = chol[SP.tril_indices(self.dim)]
        self.setParams(params)

    #####################
    # Cached
    #####################
    @cached
    def K(self):
        self._updateL()
        RV = SP.dot(self.L,self.L.T)+self.jitter*SP.eye(self.dim)
        return RV

    @cached
    def K_grad_i(self,i):
        self._updateL()
        self._updateLgrad(i)
        RV = SP.dot(self.L,self.Lgrad.T)+SP.dot(self.Lgrad,self.L.T)
        return RV[..., np.newaxis]

    ######################
    # Private functions
    ######################

    def _updateL(self):
        """
        construct the cholesky factor from hyperparameters
        """
        self.L[SP.tril_indices(self.dim)] = self.params

    def _updateLgrad(self,i):
        """
        construct the cholesky factor from hyperparameters
        """
        self.zeros[i] = 1
        self.Lgrad[SP.tril_indices(self.dim)] = self.zeros
        self.zeros[i] = 0

if __name__ == '__main__':
    n = 2
    cov = FreeFormCov(n)
    print cov.K()
    print cov.K_grad_i(0)
