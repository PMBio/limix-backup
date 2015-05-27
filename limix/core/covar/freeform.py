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
        R = self.K().copy()
        inv_diag = 1./sp.sqrt(R.diagonal())[:,sp.newaxis]
        R *= inv_diag
        R *= inv_diag.T
        return R

    # TODO: this should not be a propery
    @property
    def K_ste(self): 
        if self.getFIinv() is None:
            R = None
        else:
            R = sp.zeros((self.dim, self.dim))
            R[sp.tril_indices(self.dim)] = sp.sqrt(self.getFIinv().diagonal())
            # symmetrize
            R = R + R.T - sp.diag(R.diagonal())
        return R
        

    # THE FOLLOWING ARE BASED ON VARIANCE/CORRELATION INTERPRETABLE PARAMETRIZATION
    # TODO: reimplement those to calculate standard errors from K_ste

    #@property
    #def variance_ste(self):
    #    if self.getFIinv() is None:
    #        R = None
    #    else:
    #        R = sp.sqrt(self.getFIinv().diagonal()[:self.dim])
    #    return R

    #@property
    #def correlation_ste(self):
    #    if self.getFIinv() is None:
    #        R = None
    #    else:
    #        R = sp.zeros((self.dim, self.dim))
    #        R[sp.tril_indices(self.dim, k = -1)] = sp.sqrt(self.getFIinv().diagonal()[self.dim:])
    #        R += R.T
    #    return R

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
        # VARIANCE + CORRELATIONS
        #R1 = self.variance
        #R2 = self.correlation[sp.tril_indices(self.dim, k = -1)] 
        #R = sp.concatenate([R1,R2])

        # COVARIANCES
        R = self.K()[sp.tril_indices(self.dim)]
        return R

    # DERIVARIVE WITH RESPECT TO COVARIANCES
    def K_grad_interParam_i(self, i):
        ix, iy = sp.tril_indices(self.dim) 
        ix = ix[i]
        iy = iy[i]
        R = sp.zeros((self.dim,self.dim))
        R[ix, iy] = R[iy, ix] = 1
        return R

    # DERIVARIVE WITH RESPECT TO VARIANCES AND CORRELATIONS
    #def K_grad_interParam_i(self, i):
    #    if i < self.dim:
    #        # derivative with respect to the variance
    #        R = sp.zeros((self.dim,self.dim))
    #        R[i,:] = self.K()[i,:] / (2 * self.variance[i])
    #        R += R.T
    #    else:
    #        # derivarice with respect to a correlation
    #        ## 1. take the corresponding off diagonal element
    #        ix, iy = sp.tril_indices(self.dim, k = -1)
    #        ix = ix[i - self.dim]
    #        iy = iy[i - self.dim]
    #        ## 2. fill it with sqrt(var * var)
    #        R = sp.zeros((self.dim,self.dim))
    #        R[ix,iy] = R[iy,ix] = sp.sqrt(self.variance[ix] * self.variance[iy])
    #    return R

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
