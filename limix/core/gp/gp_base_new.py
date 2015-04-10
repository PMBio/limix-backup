import pdb
import scipy as sp
import scipy.linalg
import copy

import sys
sys.path.insert(0,'./../../..')
from limix.core.cobj import *
#from limix.core.linalg.linalg_matrix import jitChol
import limix.core.mean.mean_base
import limix.core.covar.covariance
import scipy.lib.lapack.flapack


import logging as LG


class gp(cObject):
    """
    Gaussian Process regression class for linear mean (with REML)
    y ~ N(Xb,K)
    """

    def __init__(self,covar=None,mean=None):
        """
        covar:        Covariance function
        mean:         Linear Mean function
        """
        self.covar = covar
        self.mean  = mean
        self._grad_idx = 0

    def set_grad_idx(self,value):
        """
        Set gradient index for derivatives 
        """
        self._grad_idx = value
        self.covar.set_grad_idx = value
        self.clear_cache('dKKiY','dKKiX','dKKiXB',
                            'YKiY_grad_i','YKiXB_grad_i','Areml_grad_i')

    def setParams(self,params):
        """
        Set parameters
        """
        self.covar.setParams(params['covar'])
        self.clear_cache('Areml','Areml_chol','Areml_inv',
                            'Areml_logdet','KiX','KiXB','KiY',
                            'YKiY','YKiXB','LML',
                            'dKKiY','dKKiX','dKKiXB',
                            'YKiY_grad_i','YKiXB_grad_i','Areml_grad_i'
                            'Areml_logdet_grad','YKiY_grad','YKiXB_grad','LML_grad')
        self.update_B()

    def getParams(self,params):
        """
        Get parameters
        """
        RV = {}
        RV['covar'] = self.covar.getParams()
        return rv

    #######################
    # LML terms 
    #######################
    @cached
    def Areml(self):
        return sp.dot(self.mean.X.T,self.KiX())

    #TODO: move in matrix class?
    @cached
    def Areml_chol(self):
        return LA.cholesky(self.Areml()).T

    #TODO: move in matrix class?
    @cached
    def Areml_inv(self):
        return LA.cho_solve((self.Areml_chol(),True),sp.eye(self.mean._K))

    #TODO: move in matrix class?
    @cached
    def Areml_logdet(self):
        return 2*sp.log(sp.diag(self.Areml_chol())).sum()

    # B is calculated here but cached in the mean?
    def update_B(self):
        self.mean.B = sp.dot(self.Areml_inv(),self.XKiY)

    @cached
    def KiX(self):
        return self.covar.Kinv_dot(self.mean.X)

    @cached
    def KiXB(self):
        # this can be rewritten as XKiX.Kinv_dot(self.mean.B)
        return sp.dot(self.KiX,self.mean.B)

    @cached
    def KiY(self):
        return self.covar.Kinv_dot(self.mean.Y)

    @cached
    def YKiY(self):
        return (self.mean.Y*self.KiY()).sum()

    @cached
    def YKiXB(self):
        return (self.mean.Y*self.KiXB()).sum()

    #######################
    # gradients
    #######################
    @cached
    def dKKiY(self):
        return sp.dot(self.covar.K_grad_i(),self.KiY())

    @cached
    def dKKiX(self):
        return sp.dot(self.covar.K_grad_i(),self.KiX())

    @cached
    def dKKiXB(self):
        return sp.dot(self.dKKiX(),self.mean.B)

    @cached
    def Areml_grad_i(self):
        return -sp.dot(self.KiX().T,self.dKKiX())

    @cached
    def YKiY_grad_i(self):
        return -(self.KiY()*self.dKKiY()).sum()

    @cached
    def YKiXB_grad_i(self):
        rv = -2*(self.KiY()*self.dKKiXB()).sum()
        rv+= (self.KiXB()*self.dKKiXB()).sum()
        return rv

    @cached
    def Areml_logdet_grad_i(self):
        return (self.Areml_inv()*self.Areml_grad_i()).sum()

    #######################
    # LML and gradients
    #######################

    @cached
    def LML(self):
        rv = -0.5*self.covar.logdet()
        rv -= 0.5*self.A_logdet()
        rv -= 0.5*self.YKiY()
        rv += 0.5*self.YKiB()
        return LML

    @cached
    def YKiY_grad(self):
        n_params = self.getParams()['covar'].shape[0]
        RV = SP.zeros(n_params)
        for i in range(self.n_params):
            self.set_grad_idx(i)
            RV[i] = self.YKiY_grad_i()
        return RV

    @cached
    def YKiXB_grad(self):
        n_params = self.getParams()['covar'].shape[0]
        RV = SP.zeros(n_params)
        for i in range(self.n_params):
            self.set_grad_idx(i)
            RV[i] = self.YKiXB_grad_i()
        return RV

    @cached
    def Areml_logdet_grad(self):
        n_params = self.getParams()['covar'].shape[0]
        RV = SP.zeros(n_params)
        for i in range(self.n_params):
            self.set_grad_idx(i)
            RV[i] = self.Areml_logdet_grad()
        return RV

    def LML_grad(self):
        n_params = self.getParams()['covar'].shape[0]
        RV = SP.zeros(n_params)
        for i in range(self.n_params):
            self.set_grad_idx(i)
            RV[i] = -0.5*self.covar.logdet_grad_i()
            RV[i] -= 0.5*self.A_logdet_grad_i()
            RV[i] -= 0.5*self.YKiY_grad_i()
            RV[i] += 0.5*self.YKiB_grad_i()
        return RV

    def checkGradient(self,h=1e-4,verbose=True,fun='LML'):
        """
        utility function to check the analytical gradient of
        a scalar function in the gp
        """
        f = getattr(self,fun)
        f_grad = getattr(self,fun+'_grad')
        grad_an = f_grad()
        grad_num = {}
        for key in self.params.keys():
            paramsL = self.params.copy()
            paramsR = self.params.copy()
            grad_num[key] = sp.zeros_like(self.params[key])
            e = sp.zeros(self.params[key].shape[0])
            for i in range(self.params[key].shape[0]):
                e[i] = 1
                paramsL[key]=self.params[key]-h*e
                paramsR[key]=self.params[key]+h*e
                gp.setParams(paramsL)
                lml_L = f()
                gp.setParams(paramsR)
                lml_R = f()
                grad_num[key][i] = (lml_R-lml_L)/(2*h)
                e[i] = 0
            if verbose:
                print '%s:'%key
                print abs((grad_an[key]-grad_num[key]))
                print ''

if 0:

    def predict(self,hyperparams,Xstar):
        """
        predict on Xstar
        """
        KV = self.get_covariances(hyperparams)
        Kstar = self.covar.K(hyperparams['covar'],self.X,Xstar)
        Ystar = sp.dot(Kstar.T,KV['alpha'])
        return Ystar.flatten()
        
    def get_covariances(self,hyperparams):
        """
        INPUT:
        hyperparams:  dictionary
        OUTPUT: dictionary with the fields
        K:     kernel
        Kinv:  inverse of the kernel
        L:     chol(K)
        alpha: solve(K,y)
        W:     D*Kinv * alpha*alpha^T
        """
        if self._is_cached(hyperparams):
            return self._covar_cache

        K = self.covar.K(hyperparams['covar'],self.X)
        
        if self.likelihood is not None:
            Knoise = self.likelihood.K(hyperparams['lik'],self.n)
            K += Knoise
        L = LA.cholesky(K).T# lower triangular

        alpha = LA.cho_solve((L,True),self.Y)
        Kinv = LA.cho_solve((L,True),sp.eye(L.shape[0]))
        W = self.t*Kinv - sp.dot(alpha,alpha.T)
        self._covar_cache = {}
        self._covar_cache['K'] = K
        self._covar_cache['Kinv'] = Kinv
        self._covar_cache['L'] = L
        self._covar_cache['alpha'] = alpha
        self._covar_cache['W'] = W
        self._covar_cache['hyperparams'] = copy.deepcopy(hyperparams) 
        return self._covar_cache

    def _is_cached(self,hyperparams,keys=None):
        """ check wheter model parameters are cached"""
        if self._covar_cache is None:
            return False
        if not ('hyperparams' in self._covar_cache):
            return False
        if keys==None:
            keys = hyperparams.keys()
        for key in keys:
            if (self._covar_cache['hyperparams'][key]!=hyperparams[key]).any():
                return False
        return True

    def _invalidate_cache(self):
        """ reset cache """
        self._covar_cache = None


