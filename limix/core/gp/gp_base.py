import pdb
import scipy as SP
import scipy.linalg as LA
import copy

import sys
sys.path.append('./../../..')
from limix.core.linalg.linalg_matrix import jitChol
import limix.core.likelihood.likelihood_base
import scipy.lib.lapack.flapack


import logging as LG


class GP(object):
    """
    Gaussian Process regression class. Holds all information for the GP regression to take place.

    
    """

    __slots__ = ['X','Y','n','d','t','covar','likelihood','_covar_cache','debugging']
    
    def __init__(self,covar_func=None,likelihood=None):
        """
        covar:        Covariance function
        likelihood:   Likelihood function
        x:            Inputs  [n x d]
        y:            Outputs [n x t]
        """
        self.covar = covar_func
        self.likelihood = likelihood
        self._covar_cache = None
        self.debugging = False

    def setDebugging(self,debugging):
        self.debugging = debugging
        self._invalidate_cache()
        

    def setData(self,X=None,Y=None,X_r=None):
        """
        set data

        X:    Inputs  [n x d]
        Y:    Outputs [n x t]
        """
        if X_r!=None:
            X = X_r
        
        assert X.shape[0]==Y.shape[0], 'dimensions do not match'
        self.X = X
        self.Y = Y
        self.n = X.shape[0]
        self.d = X.shape[1]
        self.t = Y.shape[1]

        self._invalidate_cache()
        
    def LML(self,hyperparams):
        """
        evalutes the log marginal likelihood for the given hyperparameters

        hyperparams
        """
        LML = self._LML_covar(hyperparams)
        return LML

    def LMLgrad(self,hyperparams):
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

    def checkGradient(self,h=1e-4,verbose=True):
        """ utility function to check the gradient of the gp """
        grad_an = self.LMLgrad()
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
                lml_L = self.LML(paramsL)
                lml_R = self.LML(paramsR)
                grad_num[key][i] = (lml_R-lml_L)/(2*h)
                e[i] = 0
            if verbose:
                print '%s:'%key
                print abs((grad_an[key]-grad_num[key]))
                print ''


