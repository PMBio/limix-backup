import pdb
import scipy as SP
import scipy.linalg as LA
import copy
import sys
from limix.core.linalg.linalg_matrix import jitChol
import limix.core.likelihood.likelihood_base

import logging as LG


class GP(object):
    """
    Gaussian Process regression class. Holds all information for the GP regression to take place.
    """

    def __init__(self,Y,K):
        """
        y:        phenotype vector
        K:        sample-to-sample covariance matrix (includes noise!)
        """
        self.covar = K
        self.setY(Y)

    def setY(self,Y):
        """
        set pheno
        """
        if Y.ndim==1: Y = Y[:,SP.newaxis]
        self.n,self.t = Y.shape
        self.Y = Y
        self.Y_has_changed = True

    def getParams(self):
        params = {}
        params['covar'] = self.covar.getParams()
        return params

    def setParams(self,params):
        self.params = params
        self.updateParams()

    def updateParams(self):
        self.covar.setParams(self.params['covar'])

    def LML(self,params=None):
        """
        evalutes the log marginal likelihood for the given hyperparameters

        hyperparams
        """
        if params is not None:
            self.setParams(params)

        KV = self._update_cache()
        alpha = KV['alpha']
        L = KV['L']

        lml_quad = 0.5 * (alpha*self.Y).sum()
        lml_det = self.t *SP.log(SP.diag(L)).sum()
        lml_const = 0.5*self.n*self.t*SP.log(2*SP.pi)
        LML = lml_quad + lml_det + lml_const
        return LML


    def LMLgrad(self,params=None):
        """
        evaluates the gradient of the log marginal likelihood for the given hyperparameters
        """
        if params is not None:
            self.setParams(params)
        KV = self._update_cache()
        W = KV['W']
        LMLgrad = SP.zeros(self.covar.n_params)
        for i in range(self.covar.n_params):
            Kd = self.covar.Kgrad_param(i)
            LMLgrad[i] = 0.5 * (W*Kd).sum()
        return {'covar':LMLgrad}


    def predict(self,Xstar):
        """
        predict on Xstar
        """
        KV = self._update_cache()
        self.covar.setXstar(Xstar)
        Kstar = self.covar.Kcross()
        Ystar = SP.dot(Kstar,KV['alpha'])
        return Ystar

    def _update_cache(self):
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
        cov_params_have_changed = self.covar.params_have_changed

        if cov_params_have_changed or self.Y_has_changed:
            K = self.covar.K()
            L = LA.cholesky(K).T# lower triangular
            Kinv = LA.cho_solve((L,True),SP.eye(L.shape[0]))
            alpha = LA.cho_solve((L,True),self.Y)
            W = self.t*Kinv - SP.dot(alpha,alpha.T)
            self._covar_cache = {}
            self._covar_cache['K'] = K
            self._covar_cache['Kinv'] = Kinv
            self._covar_cache['L'] = L
            self._covar_cache['alpha'] = alpha
            self._covar_cache['W'] = W

        return self._covar_cache


    def checkGradient(self,h=1e-6,verbose=True):
        """ utility function to check the gradient of the gp """
        grad_an = self.LMLgrad()
        grad_num = {}
        params0 = self.params.copy()
        for key in list(self.params.keys()):
            paramsL = params0.copy()
            paramsR = params0.copy()
            grad_num[key] = SP.zeros_like(self.params[key])
            e = SP.zeros(self.params[key].shape[0])
            for i in range(self.params[key].shape[0]):
                e[i] = 1
                paramsL[key]=params0[key]-h*e
                paramsR[key]=params0[key]+h*e
                lml_L = self.LML(paramsL)
                lml_R = self.LML(paramsR)
                grad_num[key][i] = (lml_R-lml_L)/(2*h)
                e[i] = 0
            if verbose:
                print(('%s:'%key))
                print((abs(grad_an[key]-grad_num[key])))
                print('')
        self.setParams(params0)
