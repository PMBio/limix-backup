from covar_base import Covariance
import pdb
import numpy as np
import scipy as sp
from limix.core.type.cached import Cached, cached

class KronCov(Covariance):
    """
    Kronecker product between two covariances
    """

    def __init__(self, C, R):
        """
        Args:
            C:     column LIMIX covariance 
            R:     row numpy covariance matrix 
        """
        Covariance.__init__(self)
        self.dim = C.dim * R.shape[0]
        self._C = C
        self._R = R
        C.register(self.clear_all)

    #####################
    # Properties
    #####################
    @property
    def C(self):
        return self._C

    @property
    def R(self):
        return self._R

    #####################
    # Params handling
    #####################
    def setParams(self,params):
        self.C.setParams(params)

    def getParams(self):
        return self.C.getParams()

    def getNumberParams(self):
        return self.C.getNumberParams() 

    ####################
    # Predictions
    ####################
    @property
    def use_to_predict(self):
        return self.C.use_to_predict

    @use_to_predict.setter
    def use_to_predict(self,value):
        self.C.use_to_predict = value

    #####################
    # Cached
    #####################
    @cached('covar_base')
    def K(self):
        return sp.kron(self.C.K(), self.R)

    @cached('covar_base')
    def Kcross(self):
        #TODO: create support for Rcross
        # it woud be simply:
        #return sp.kron(self.C.K(), self.Rcross)
        pass

    @cached('covar_base')
    def K_grad_i(self,i):
        return sp.kron(self.C.K_grad_i(i), self.R)

    @cached('covar_base')
    def K_hess_i_j(self, i, j):
        return sp.kron(self.C.K_hess_i_j(i, j), self.R)

    def _calcNumberParams(self):
        self.n_params = self.C.getNumberParams() 
        return self.n_params

    ####################
    # Interpretable Params
    ####################
    def getInterParams(self):
        return self.C.getInterParams()

    def K_grad_interParam_i(self,i):
        return sp.kron(self.C.K_grad_interParam_i(i), self.R)

    def setFIinv(self, value):
        self.C.setFIinv(value)

if __name__=='__main__':
    from limix.core.covar import FreeFormCov
    from limix.utils.preprocess import covar_rescale

    # define row caoriance
    dim_r = 10
    X = sp.rand(dim_r, dim_r)
    R = covar_rescale(sp.dot(X,X.T))

    # define col covariance
    dim_c = 3
    C = FreeFormCov(dim_c)

    cov = KronCov(C, R) 
    cov.setRandomParams()
    print cov.K()
    print cov.K_grad_i(0)

