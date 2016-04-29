from .covar_base import Covariance
import pdb
import numpy as np
import scipy as sp
from hcache import Cached, cached

class KronCov(Covariance):
    """
    Kronecker product between two covariances
    """

    def __init__(self, C, R, Iok=None):
        """
        Args:
            C:     column LIMIX covariance 
            R:     row numpy covariance matrix 
        """
        Covariance.__init__(self)
        self._C = C
        self._R = R
        self.dim = C.dim * R.shape[0]
        self.Iok = Iok 
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

    @property
    def Iok(self):
        return self._Iok

    #####################
    # Setters
    #####################
    @Iok.setter
    def Iok(self, value):
        if value is not None:
            assert len(value.shape)==1, 'must be a 1-dimensioanal array'
            assert value.shape[0]==self.dim, 'Dimension mismatch'
            self.dim = value.sum()
        self._Iok = value

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
        R = sp.kron(self.C.K(), self.R)
        if self.Iok is not None:
            R = R[self.Iok][:, self.Iok]
        return R

    @cached('covar_base')
    def Kcross(self):
        #TODO: create support for Rcross
        # it woud be simply:
        #return sp.kron(self.C.K(), self.Rcross)
        pass

    @cached('covar_base')
    def K_grad_i(self,i):
        R = sp.kron(self.C.K_grad_i(i), self.R)
        if self.Iok is not None:
            R = R[self.Iok][:, self.Iok]
        return R

    @cached('covar_base')
    def K_hess_i_j(self, i, j):
        R = sp.kron(self.C.K_hess_i_j(i, j), self.R)
        if self.Iok is not None:
            R = R[self.Iok][:, self.Iok]
        return R

    def _calcNumberParams(self):
        self.n_params = self.C.getNumberParams() 
        return self.n_params

    ####################
    # Interpretable Params
    ####################
    def getInterParams(self):
        return self.C.getInterParams()

    def K_grad_interParam_i(self,i):
        R = sp.kron(self.C.K_grad_interParam_i(i), self.R)
        if self.Iok is not None:
            R = R[self.Iok][:,self.Iok]
        return R

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
    print((cov.K()))
    print((cov.K_grad_i(0)))

