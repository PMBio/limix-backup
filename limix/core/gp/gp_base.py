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
        return LA.cho_solve((self.Areml_chol(),True),SP.eye(self.mean._K))

    #TODO: move in matrix class?
    @cached
    def Areml_logdet(self):
        return 2*SP.log(SP.diag(self.Areml_chol())).sum()

    # B is calculated here but cached in the mean?
    def update_B(self):
        self.mean.B = SP.dot(self.Areml_inv(),self,XKiY)

    @cached
    def KiXB(self):
        # TODO
        pass

    @cached
    def KiY(self):
        return self.covar.Kinv_dot(self.mean.Y)

    @cached
    def YKiY(self):
        return (self.mean.Y*self.KiY()).sum()

    @cached
    def YKiXB(self):
        # TODO
        pass

    #######################
    # gradients
    #######################
    #TODO caching?
    def dKKiY(self,i):
        dK = self.covar.Kgrad_param(i)
        return SP.dot(dK,self.KiY())

    def dKKiX(self,i):
        dK = self.covar.Kgrad_param(i)
        return SP.dot(dK,self.KiX())

    #TODO: in c principle these guys should have the key as well
    def YKiY_grad(self,i):
        return -(self.KiY()*self.dKKiY(i)).sum()

    def YKiXB_grad(self,i):
        pass

    def Areml_logdet_grad(self,i):
        pass

    #######################
    # LML and gradients
    #######################

    @cached
    def LML(self):
        rv = -0.5*self.covar.logdet()
        rv -= 0.5*self.A_logdet()
        rv -= 0.5*self.yKiy()
        rv += 0.5*self.yKiB()
        return LML

    def LML_grad(self,hyperparams):
        """
        evaluates the gradient of the log marginal likelihood for the given hyperparameters
        """
        RV = {}
        # gradient with respect to hyperparameters
        RV.update(self._LMLgrad_covar(hyperparams))
        if self.likelihood != None:
            # gradient with respect to noise parameters
            RV.update(self._LMLgrad_lik(hyperparams))
        return RV

    def _LML_covar(self,hyperparams):
        """
        log marginal likelihood
        """
        try:
            KV = self.get_covariances(hyperparams)
        except LA.LinAlgError:
            LG.error('linalg exception in _LML_covar')
            return 1E6
        
        alpha = KV['alpha']
        L = KV['L']
        
        lml_quad = 0.5 * (alpha*self.Y).sum()
        lml_det = self.t *SP.log(SP.diag(L)).sum()
        lml_const = 0.5*self.n*self.t*SP.log(2*SP.pi)
        LML = lml_quad + lml_det + lml_const
        return LML

    def _LMLgrad_covar(self,hyperparams):
        """
        evaluates the gradient of the log marginal likelihood with respect to the
        hyperparameters of the covariance function
        """
        logtheta = hyperparams['covar']

        try:
            KV = self.get_covariances(hyperparams)
        except LA.LinAlgError:
            LG.error('linalg exception in _LMLgrad_covar')
            return {'covar':SP.zeros(len(logtheta))}
        
        W = KV['W']
        n_theta = len(logtheta)
        LMLgrad = SP.zeros(len(logtheta))
        for i in xrange(n_theta):
            Kd = self.covar.Kgrad_theta(hyperparams['covar'],self.X,i)
            LMLgrad[i] = 0.5 * (W*Kd).sum()
        return {'covar':LMLgrad}

    def _LMLgrad_lik(self,hyperparams):
        """
        evaluates the gradient of the log marginal likelihood with respect to
        the hyperparameters of the likelihood function
        """
        logtheta = hyperparams['lik']
        KV = self._covar_cache
        W = KV['W']
        n_theta = len(logtheta)
        LMLgrad = SP.zeros(len(logtheta))
        for i in xrange(n_theta):
            Kd = self.likelihood.Kgrad_theta(hyperparams['lik'],self.n,i)
            LMLgrad[i] = 0.5 * (W*Kd).sum()
        return {'lik':LMLgrad}


    def predict(self,hyperparams,Xstar):
        """
        predict on Xstar
        """
        KV = self.get_covariances(hyperparams)
        Kstar = self.covar.K(hyperparams['covar'],self.X,Xstar)
        Ystar = SP.dot(Kstar.T,KV['alpha'])
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
        Kinv = LA.cho_solve((L,True),SP.eye(L.shape[0]))
        W = self.t*Kinv - SP.dot(alpha,alpha.T)
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
            grad_num[key] = SP.zeros_like(self.params[key])
            e = SP.zeros(self.params[key].shape[0])
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

