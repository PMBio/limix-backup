import numpy as np
import scipy as sp
import scipy.linalg as LA
from covar_base import Covariance
from limix.core.type.cached import cached
import pdb

import logging as LG

class FreeFormCov(Covariance):
    """
    freeform covariance function
    """
    def __init__(self,dim,jitter=1e-4):
        Covariance.__init__(self, dim)
        self.n_params = int(dim * (dim + 1.) / 2.)
        self.dim = dim
        self.params = sp.zeros(self.n_params)

        self.L = sp.zeros((self.dim,self.dim))
        self.Lgrad = sp.zeros((self.dim,self.dim))
        self.zeros = sp.zeros(self.n_params)
        self.set_jitter(jitter)

    #####################
    # Properties
    #####################
    @property
    def variance(self):
        return self.K().diagonal() 

    @property
    def correlation(self):
        R = self.K()
        inv_diag = 1./sp.sqrt(R.diagonal())[:,sp.newaxis]
        R /= inv_diag
        R /= inv_diag.T
        return R

    @property
    def variance_ste(self):
        LG.critical("Implement variance_ste")

    @property
    def correlation_ste(self):
        LG.critical("Implement correlation_ste")

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
        params = chol[sp.tril_indices(self.dim)]
        self.setParams(params)

    #####################
    # Cached
    #####################
    @cached
    def K(self):
        self._updateL()
        RV = sp.dot(self.L,self.L.T)+self.jitter*sp.eye(self.dim)
        return RV

    @cached
    def K_grad_i(self,i):
        self._updateL()
        self._updateLgrad(i)
        RV = sp.dot(self.L,self.Lgrad.T)+sp.dot(self.Lgrad,self.L.T)
        return RV

    ####################
    # Interpretable Params
    ####################
    def getInterParams(self):
        R1 = self.variance()
        R2 = self.correlation()[sp.tril_indices(self.dim)] 
        R = sp.concatenate([R1,R2])
        return R

    def K_grad_interParam_i(self,i):
        LG.critical("implement K_grad_interParam_i")

    ######################
    # Private functions
    ######################

    def _updateL(self):
        """
        construct the cholesky factor from hyperparameters
        """
        self.L[sp.tril_indices(self.dim)] = self.params

    def _updateLgrad(self,i):
        """
        construct the cholesky factor from hyperparameters
        """
        self.zeros[i] = 1
        self.Lgrad[sp.tril_indices(self.dim)] = self.zeros
        self.zeros[i] = 0

if __name__ == '__main__':
    n = 2
    cov = FreeFormCov(n)
    print cov.K()
    print cov.K_grad_i(0)
