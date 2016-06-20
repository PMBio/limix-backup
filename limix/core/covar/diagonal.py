import numpy as np
import scipy as sp
import scipy.linalg as LA
from .covar_base import Covariance
from hcache import cached
import pdb

import logging as LG

class DiagonalCov(Covariance):
    """
    Diagonal covairance matrix
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
        self.set_jitter(jitter)

    #####################
    # Properties
    #####################
    @property
    def variance(self):
        return sp.exp(self.params) 

    @property
    def variance_ste(self):
        #TODO
        pass

    @property
    def X(self):
        return sp.sqrt(self.K())

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
    def setParams(self, params):
        if not self._K_act and len(params) > 0:
            raise ValueError("Trying to set a parameter via setParams that "
                             "is not active.")
        if self._K_act:
            self.params[:] = params
            self.clear_all()

    def getParams(self):
        if not self._K_act:
            return np.array([])
        return self.params

    def getNumberParams(self):
        return int(self._K_act) * self.n_params

    def _calcNumberParams(self):
        self.n_params = self.dim

    def set_jitter(self,value):
        self.jitter = value

    def setCovariance(self,cov):
        """ set hyperparameters from given covariance """
        self.setParams(sp.log(sp.diagonal(cov)))

    #####################
    # Cached
    #####################
    @cached('covar_base')
    def K(self):
        return sp.diag(self.variance)

    @cached('covar_base')
    def K_grad_i(self,i):
        if not self._K_act:
            raise ValueError("Trying to retrieve the gradient over a "
                             "parameter that is inactive.")
        RV = sp.zeros((self.dim, self.dim)) 
        RV[i, i] = self.variance[i]
        return RV

    @cached
    def K_hess_i_j(self, i, j):
        if not self._K_act:
            raise ValueError("Trying to retrieve the gradient over a "
                             "parameter that is inactive.")
        RV = sp.zeros((self.dim, self.dim)) 
        if i==j:    RV[i, i] = self.variance[i]
        return RV

    def K_ste(self):
        #TODO
        #if self.getFIinv() is None:
        #    R = None
        #else:
        #    R = sp.zeros((self.dim, self.dim))
        #    R[sp.tril_indices(self.dim)] = sp.sqrt(self.getFIinv().diagonal())
        #    # symmetrize
        #    R = R + R.T - sp.diag(R.diagonal())
        #return R
        pass

    ####################
    # Interpretable Params
    ####################
    def getInterParams(self):
        return self.variance

    # DERIVARIVE WITH RESPECT TO COVARIANCES
    def K_grad_interParam_i(self, i):
        R = sp.zeros((self.dim,self.dim))
        R[i, i] = 1
        return R

