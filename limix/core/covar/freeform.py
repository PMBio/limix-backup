import numpy as np
import scipy as sp
import scipy.linalg as LA
from covar_base import Covariance
from limix.core.type.cached import cached
import pdb

import logging as LG

class FreeFormCov(Covariance):
    """
    General semi-definite positive matrix with no contraints.
    A free-form covariance matrix of dimension d has 1/2 * d * (d + 1) params
    """
    def __init__(self, dim, jitter=1e-4):
        """
        Args:
            dim:        dimension of the free-form covariance
            jitter:     extent of diagonal offset which is added for numerical stability
                        (default value: 1e-4)
        """
        Covariance.__init__(self, dim)
        self._K_act = True
        self._calcNumberParams()
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

    @property
    def variance_ste(self):
        if self.getFIinv() is None:
            R = None
        else:
            R = self.K_ste().diagonal()
        # IN A VARIANCE / CORRELATION PARAMETRIZATION
        #if self.getFIinv() is None:
        #   R = None
        #else:
        #   R = sp.sqrt(self.getFIinv().diagonal()[:self.dim])
        return R

    @property
    def correlation_ste(self):
        if self.getFIinv() is None:
            R = None
        else:
            idx_M = sp.zeros((self.dim,self.dim))
            idx_M[sp.tril_indices(self.dim)] = sp.arange( int( 0.5 * self.dim * (self.dim + 1) ) )
            R = sp.zeros(idx_M)
            for i in range(self.dim):
                for j in range(0,self.dim):
                    ij = idx_M[i,j] # index of cov_ij_ste from fisher
                    ii = idx_M[i,i] # index of cov_ii_ste from fisher
                    jj = idx_M[j,j] # index of cov_jj_ste from fisher
                    #TODO: complete

        # IN A VARIANCE / CORRELATION PARAMETRIZATION
        #if self.getFIinv() is None:
        #    R = None
        #else:
        #    R = sp.zeros((self.dim, self.dim))
        #    R[sp.tril_indices(self.dim, k = -1)] = sp.sqrt(self.getFIinv().diagonal()[self.dim:])
        #    R += R.T
        return R

    #####################
    # Activation handling
    #####################
    @property
    def act_K(self):
        return self._K_act

    @act_K.setter
    def act_K(self, act):
        self._K_act = bool(act)
        self._notify()

    #####################
    # Params handling
    #####################
    def setParams(self, params, notify=True):
        if not self._K_act and len(params) > 0:
            raise ValueError("Trying to set a parameter via setParams that "
                             "is not active.")
        if self._K_act:
            self.params[:] = params
            # self.clear_all()
            self.clear_cache('default')
            if notify:
                self._notify()

    def getParams(self):
        if not self._K_act:
            return np.array([])
        return self.params

    def getNumberParams(self):
        return int(self._K_act) * self.n_params

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
        if not self._K_act:
            raise ValueError("Trying to retrieve the gradient over a "
                             "parameter that is inactive.")

        self._updateL()
        self._updateLgrad(i)
        RV = sp.dot(self.L,self.Lgrad.T)+sp.dot(self.Lgrad,self.L.T)
        return RV

    def K_ste(self):
        if self.getFIinv() is None:
            R = None
        else:
            R = sp.zeros((self.dim, self.dim))
            R[sp.tril_indices(self.dim)] = sp.sqrt(self.getFIinv().diagonal())
            # symmetrize
            R = R + R.T - sp.diag(R.diagonal())
        return R

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
