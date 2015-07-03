import sys
from limix.core.type.observed import Observed
from limix.core.type.cached import Cached, cached
from limix.utils.eigen import *
import scipy as sp
import scipy.linalg as LA
import warnings

import logging as LG

class Covariance(Cached, Observed):
    """
    abstract super class for all implementations of covariance functions
    """
    def __init__(self,dim=None):
        Cached.__init__(self)
        if dim is not None:
            self.initialize(dim)

    def initialize(self,dim):
        self.dim = dim
        self._use_to_predict = False

    def clear_all(self):
        # TODO: define groups for those
        self.clear_cache('K','Kcross','K_grad_i',
                          'logdet','logdet_grad_i',
                          'inv','chol','S','U','USi2')

    #######################
    # Param Handling
    #######################
    # def getParams(self):
    #     return self.params
    #
    # def setParams(self,params):
    #     self.params = params
    #     self.clear_all()
    #     self._notify()

    def setRandomParams(self):
        """
        set random hyperparameters
        """
        params = sp.randn(self.getNumberParams())
        self.setParams(params)

    def setCovariance(self,cov):
        """
        set hyperparameters from given covariance
        """
        warnings.warn('not implemented')


    def perturbParams(self,pertSize=1e-3):
        """
        slightly perturbs the values of the parameters
        """
        params = self.getParams()
        self.setParams(params+pertSize*sp.randn(params.shape[0]))

    def _calcNumberParams(self):
        """
        calculates the number of parameters
        """
        warnings.warn('not implemented')

    ####################################
    # cached
    ####################################
    @cached
    def K(self):
        """
        evaluates the kernel for given hyperparameters theta and inputs X
        """
        LG.critical("implement K")
        print("%s: Function K not yet implemented"%(self.__class__))
        return None

    @cached
    def K_grad_i(self,i):
        """
        partial derivative with repspect to the i-th hyperparamter theta[i]
        """
        LG.critical("implement K_grad_i")
        print("%s: Function K not yet implemented"%(self.__class__))

    def solve(self,M):
        return LA.cho_solve((self.chol(),True),M)

    @cached
    def chol(self):
        return LA.cholesky(self.K()).T

    @cached
    def inv(self):
        return self.solve(sp.eye(self.dim))

    @cached
    def logdet(self):
        return 2*sp.log(sp.diag(self.chol())).sum()

    @cached
    def logdet_grad_i(self,i):
        return self.solve(self.K_grad_i(i)).diagonal().sum()

    @cached
    def S(self):
        RV,U = LA.eigh(self.K())
        self.fill_cache('U',U)
        return RV

    @cached
    def U(self):
        S,RV = LA.eigh(self.K())
        self.fill_cache('S',S)
        return RV

    @cached
    def USi2(self):
        # U * S**(-1/2)
        return self.U()*(self.S()**(-0.5))

    ###########################
    # The following methods are deprecated
    ############################

    #@cached
    #def Sgrad(self,i):
    #    return dS_dti(self.Kgrad_param(i),U=self.U())

    #@cached
    #def Ugrad(self,i):
    #    return dU_dti(self.Kgrad_param(i),U=self.U(),S=self.S())

    #@cached
    #def USi2grad(self,i):
    #    # dU * S**(-1/2) + U * (-1/2 S**(-3/2) dS)
    #    Si2grad = -0.5*self.S()**(-1.5)*self.Sgrad(i)
    #    return self.Ugrad(i)*(self.S()**(-0.5)) + self.U()*Si2grad

    ###########################
    # Predictions
    ###########################
    @property
    def use_to_predict(self):
        return self._use_to_predict

    @use_to_predict.setter
    def use_to_predict(self,value):
        LG.critical("implement use_to_predict")
        print("%s: Function use_to_predict not yet implemented"%(self.__class__))

    @cached
    def Kcross(self):
        LG.critical("implement Kcross")
        print("%s: Function Kcross not yet implemented"%(self.__class__))

    ####################
    # Interpretable Params, Fisher information Matrix, std errors
    ####################
    def getInterParams(self):
        return self.getParams()

    def K_grad_interParam_i(self,i):
        return K_grad_i_interParams(self,i)

    def getFisherInf(self):
        n_params = self.getNumberParams()
        R = sp.zeros((n_params,n_params))
        for m in range(n_params):
            for n in range(n_params):
                DnK = self.K_grad_interParam_i(m)
                DmK = self.K_grad_interParam_i(n)
                KiDnK = self.solve(DnK)
                KiDmK = self.solve(DmK)
                R[m,n] = 0.5*(KiDnK*KiDmK).sum()
        return R

    def setFIinv(self, value):
        self._FIinv = value

    def getFIinv(self):
        return self._FIinv

    ############################
    # Debugging
    ############################

    def Kgrad_param_num(self,i,h=1e-4):
        """
        check discrepancies between numerical and analytical gradients
        """
        params = self.getParams()
        e = sp.zeros_like(params); e[i] = 1
        self.setParams(params-h*e)
        C_L = self.K()
        self.setParams(params+h*e)
        C_R = self.K()
        self.setParams(params)
        RV = (C_R-C_L)/(2*h)
        return RV
