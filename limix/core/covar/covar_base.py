import sys
from limix.core.type.observed import Observed
from hcache import Cached, cached
from limix.utils.eigen import *
import scipy as sp
import scipy.linalg as LA
import scipy.sparse.linalg as sla
import warnings

import logging as LG

class Covariance(Cached, Observed):
    """
    abstract super class for all implementations of covariance functions
    """
    def __init__(self, dim=None, nIterMC=30):
        Cached.__init__(self)
        self._nIterMC = nIterMC
        self._reuse = True
        self._KiZo = None
        self._tol = 1e-6
        if dim is not None:
            self.initialize(dim)

    def initialize(self,dim):
        self.dim = dim
        self._use_to_predict = False

    def clear_all(self):
        self.clear_cache('covar_base')
        self._notify()

    #######################
    # Param Handling
    #######################

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


    def perturbParams(self, pertSize=1e-3):
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

    ###################################
    # Non-cached methods
    ###################################
    def dot(self, M):
        return sp.dot(self.K(), M)

    def solve(self,M):
        return LA.cho_solve((self.chol(),True),M)

    def solve_ls(self, M, M0=None):
        if M0 is None:    M0 = 1E-3 * sp.randn(*M.shape)
        def veKvei(m):
            _M = m.reshape(M.shape, order='F')
            return self.dot(_M).reshape(M.size, order='F')
        Kx_O = sla.LinearOperator((M.size, M.size), matvec=veKvei, rmatvec=veKvei, dtype='float64')
        # vectorize
        m  = M.reshape(M.size, order='F')
        m0 = M0.reshape(M0.size, order='F')
        r, _ = sla.cgs(Kx_O, m, x0=m0, tol=self._tol)
        return r.reshape(M.shape, order='F')

    ####################################
    # cached
    ####################################
    @cached('covar_base')
    def K(self):
        """
        evaluates the kernel for given hyperparameters theta and inputs X
        """
        LG.critical("implement K")
        print(("%s: Function K not yet implemented"%(self.__class__)))
        return None

    @cached('covar_base')
    def K_grad_i(self,i):
        """
        partial derivative with repspect to the i-th hyperparamter theta[i]
        """
        LG.critical("implement K_grad_i")
        print(("%s: Function K not yet implemented"%(self.__class__)))

    @cached('covar_base')
    def K_hess_i_j(self,i,j):
        LG.critical("implement K_hess_i_j")
        print(("%s: Function Khess not yet implemented"%(self.__class__)))

    @cached('covar_base')
    def chol(self):
        return LA.cholesky(self.K()).T

    @cached('covar_base')
    def inv(self):
        return self.solve(sp.eye(self.dim))

    @cached('covar_base')
    def logdet(self):
        return 2*sp.log(sp.diag(self.chol())).sum()

    @cached('covar_base')
    def logdet_grad_i(self,i):
        return self.solve(self.K_grad_i(i)).diagonal().sum()

    @cached('covar_base')
    def S(self):
        RV,U = LA.eigh(self.K())
        self.fill_cache('U',U)
        return RV

    @cached('covar_base')
    def U(self):
        S,RV = LA.eigh(self.K())
        self.fill_cache('S',S)
        return RV

    @cached('covar_base')
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
    # Monte Carlo methods
    ###########################
    def resample(self):
        self.clear_cache('Z')
        self._notify()

    @cached('Z')
    def Z(self):
        r = sp.randn(self.dim, self._nIterMC)
        # norm Z to improve convergence
        norm = sp.sqrt(self.dim / (float(self._nIterMC) * (r**2).sum(0)))
        return norm * r

    @cached(['covar_base', 'Z'])
    def DKZ(self):
        R = sp.zeros((self.dim, self._nIterMC, self.getNumberParams()))
        for i in range(R.shape[2]):
            R[:, :, i] = sp.dot(self.K_grad_i(i), self.Z())
        return R

    @cached(['covar_base', 'Z'])
    def DDKZ(self):
        R = sp.zeros((self.dim, self._nIterMC, self.getNumberParams(), self.getNumberParams()))
        for i in range(R.shape[2]):
            R[:, :, i, i] = sp.dot(self.K_hess_i_j(i, i), self.Z())
            for j in range(i):
                R[:, :, i, j] = sp.dot(self.K_hess_i_j(i, j), self.Z())
                R[:, :, j, i] = R[:, :, i, j]
        return R

    @cached(['covar_base', 'Z'])
    def KiZ(self):
        R = self.solve_ls(self.Z(), M0=self._KiZo)
        if self._reuse:     self._KiZo = R
        return R

    @cached(['covar_base', 'Z'])
    def sample_logdet_grad_i(self, i):
        DiKZ = sp.dot(self.K_grad_i(i), self.Z())
        return (DiKZ * self.KiZ()).sum()

    @cached(['covar_base', 'Z'])
    def sample_logdet_grad(self):
        return (self.DKZ() * self.KiZ()[:, :, sp.newaxis]).sum(axis=(0, 1))

    @cached(['covar_base', 'Z'])
    def sample_trKiDDK(self):
        return (self.DDKZ() * self.KiZ()[:, :, sp.newaxis, sp.newaxis]).sum(axis=(0, 1))


    ###########################
    # Predictions
    ###########################
    @property
    def use_to_predict(self):
        return self._use_to_predict

    @use_to_predict.setter
    def use_to_predict(self,value):
        LG.critical("implement use_to_predict")
        print(("%s: Function use_to_predict not yet implemented"%(self.__class__)))

    @cached
    def Kcross(self):
        LG.critical("implement Kcross")
        print(("%s: Function Kcross not yet implemented"%(self.__class__)))

    ####################
    # Interpretable Params, Fisher information Matrix, std errors
    ####################
    def getInterParams(self):
        return self.getParams()

    def K_grad_interParam_i(self,i):
        LG.critical("implement K_grad_interParam_i")
        print(("%s: Function K_grad_interParam_i not yet implemented"%(self.__class__)))

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
