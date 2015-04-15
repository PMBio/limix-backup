import numpy as np
import scipy as SP
import scipy.linalg as LA
from covariance import covariance
import pdb

class freeform(covariance):
    """
    freeform covariance function
    """
    def __init__(self,dim,jitter=1e-4):
        """
        initialization
        """
        covariance.__init__(self)
        self.n_params = int(dim * (dim + 1.) / 2.)
        self.dim = dim
        self.params = SP.zeros(self.n_params)

        self.L = SP.zeros((self.dim,self.dim))
        self.Lgrad = SP.zeros((self.dim,self.dim))
        self.zeros = SP.zeros(self.n_params)
        self.set_jitter(jitter)

    def setParams(self,params):
        """
        set hyperparameters
        """
        self.params = params
        # this is to make the old implementation work
        self.params_have_changed = True

    def getParams(self):
        return self.params

    def getNumberParams(self):
        """
        return the number of hyperparameters
        """
        return self.n_params

    def set_jitter(self,value):
        self.jitter = value

    def setCovariance(self,cov):
        """
        set hyperparameters from given covariance
        """
        chol = LA.cholesky(cov,lower=True)
        params = chol[SP.tril_indices(self.dim)]
        self.setParams(params)

    def K(self):
        """
        evaluates the kernel for given hyperparameters theta
        """
        self._updateL()
        RV = SP.dot(self.L,self.L.T)+self.jitter*SP.eye(self.dim)
        return RV

    def Kgrad_param(self,i):
        """
        partial derivative with repspect to the i-th hyperparamter theta[i]
        """
        self._updateL()
        self._updateLgrad(i)
        RV = SP.dot(self.L,self.Lgrad.T)+SP.dot(self.Lgrad,self.L.T)
        return RV[..., np.newaxis]

    def _calcNumberParams(self):
        """
        calculates the number of parameters
        """
        self.n_params = int(0.5*self.dim*(self.dim+1))

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
    cov = freeform(n)
    print cov.K()
    print cov.Kgrad_param(0)
