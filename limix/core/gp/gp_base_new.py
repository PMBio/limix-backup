import pdb
import scipy as sp
import scipy.linalg
import copy
import sys
sys.path.insert(0,'./../../..')
from limix.core.cobj import *
import limix.core.mean.mean_base
import limix.core.covar.covariance

class gp(cObject):
    """
    Gaussian Process regression class for linear mean (with REML)
    y ~ N(Fb,K)
    """

    def __init__(self,mean=None,covar=None):
        """
        covar:        Covariance function
        mean:         Linear Mean function
        """
        self.covar = covar
        self.mean  = mean
        self.set_grad_idx(0)
        self.clear_all()
        self.update_B()

    def clear_all(self):
        self.clear_Areml()
        self.clear_lml_terms()
        self.clear_lmlgrad_terms_i()
        self.clear_lmlgrad_terms()

    def clear_Areml(self):
        self.clear_cache('Areml','Areml_chol','Areml_inv','Areml_logdet')

    def clear_lml_terms(self):
        self.clear_cache('KiF','YKiF','KiFB','KiY','YKiY','YKiFB','LML')

    def clear_lmlgrad_terms_i(self):
        self.clear_cache('dKKiY','dKKiF','dKKiFB','Areml_grad_i',
                            'YKiY_grad_i','YKiFB_grad_i',
                            'Areml_logdet_grad_i')

    def clear_lmlgrad_terms(self):
        self.clear_cache('YKiY_grad','YKiFB_grad','Areml_logdet_grad','LML_grad')

    def set_grad_idx(self,value):
        """
        Set gradient index for derivatives 
        """
        self._grad_idx = value
        self.covar.set_grad_idx(value)
        self.clear_lmlgrad_terms_i()

    def setParams(self,params):
        """
        Set parameters
        """
        self.covar.setParams(params['covar'])
        self.clear_all()
        self.update_B()

    def getParams(self):
        """
        Get parameters
        """
        RV = {}
        RV['covar'] = self.covar.getParams()
        return RV

    #######################
    # LML terms 
    #######################
    @cached
    def Areml(self):
        return sp.dot(self.mean.F.T,self.KiF())

    #TODO: move in matrix class?
    @cached
    def Areml_chol(self):
        return sp.linalg.cholesky(self.Areml()).T

    #TODO: move in matrix class?
    @cached
    def Areml_inv(self):
        return sp.linalg.cho_solve((self.Areml_chol(),True),sp.eye(self.mean._K))

    #TODO: move in matrix class?
    @cached
    def Areml_logdet(self):
        return 2*sp.log(sp.diag(self.Areml_chol())).sum()

    @cached
    def KiF(self):
        return self.covar.Kinv_dot(self.mean.F)

    @cached
    def YKiF(self):
        return sp.dot(self.mean.Y.T,self.KiF())

    # B is calculated here but cached in the mean?
    def update_B(self):
        self.mean.B = sp.dot(self.Areml_inv(),self.YKiF().T)

    @cached
    def KiFB(self):
        # this can be rewritten as FKiF.Kinv_dot(self.mean.B)
        return sp.dot(self.KiF(),self.mean.B)

    @cached
    def KiY(self):
        return self.covar.Kinv_dot(self.mean.Y)

    @cached
    def YKiY(self):
        return (self.mean.Y*self.KiY()).sum()

    @cached
    def YKiFB(self):
        return (self.mean.Y*self.KiFB()).sum()

    #######################
    # gradients
    #######################
    @cached
    def dKKiY(self):
        return sp.dot(self.covar.K_grad_i(),self.KiY())

    @cached
    def dKKiF(self):
        return sp.dot(self.covar.K_grad_i(),self.KiF())

    @cached
    def dKKiFB(self):
        return sp.dot(self.dKKiF(),self.mean.B)

    @cached
    def Areml_grad_i(self):
        return -sp.dot(self.KiF().T,self.dKKiF())

    @cached
    def YKiY_grad_i(self):
        return -(self.KiY()*self.dKKiY()).sum()

    @cached
    def YKiFB_grad_i(self):
        rv = -2*(self.KiY()*self.dKKiFB()).sum()
        rv+= (self.KiFB()*self.dKKiFB()).sum()
        return rv

    @cached
    def Areml_logdet_grad_i(self):
        return (self.Areml_inv()*self.Areml_grad_i()).sum()

    #######################
    # LML and gradients
    #######################

    @cached
    def LML(self):
        #const term to add?
        rv = -0.5*self.covar.logdet()
        rv -= 0.5*self.Areml_logdet()
        rv -= 0.5*self.YKiY()
        rv += 0.5*self.YKiFB()
        return rv 

    @cached
    def YKiY_grad(self):
        n_params = self.getParams()['covar'].shape[0]
        RV = {'covar': sp.zeros(n_params)}
        for i in range(n_params):
            self.set_grad_idx(i)
            RV['covar'][i] = self.YKiY_grad_i()
        return RV

    @cached
    def YKiFB_grad(self):
        n_params = self.getParams()['covar'].shape[0]
        RV = {'covar': sp.zeros(n_params)}
        for i in range(n_params):
            self.set_grad_idx(i)
            RV['covar'][i] = self.YKiFB_grad_i()
        return RV

    @cached
    def Areml_logdet_grad(self):
        n_params = self.getParams()['covar'].shape[0]
        RV = {'covar': sp.zeros(n_params)}
        for i in range(n_params):
            self.set_grad_idx(i)
            RV['covar'][i] = self.Areml_logdet_grad_i()
        return RV

    def LML_grad(self):
        n_params = self.getParams()['covar'].shape[0]
        RV = {'covar': sp.zeros(n_params)}
        for i in range(n_params):
            self.set_grad_idx(i)
            RV['covar'][i] = -0.5*self.covar.logdet_grad_i()
            RV['covar'][i] -= 0.5*self.Areml_logdet_grad_i()
            RV['covar'][i] -= 0.5*self.YKiY_grad_i()
            RV['covar'][i] += 0.5*self.YKiFB_grad_i()
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
        params = self.getParams()
        for key in params.keys():
            paramsL = params.copy()
            paramsR = params.copy()
            grad_num[key] = sp.zeros_like(params[key])
            e = sp.zeros(params[key].shape[0])
            for i in range(params[key].shape[0]):
                e[i] = 1
                paramsL[key]=params[key]-h*e
                paramsR[key]=params[key]+h*e
                self.setParams(paramsL)
                lml_L = f()
                self.setParams(paramsR)
                lml_R = f()
                grad_num[key][i] = (lml_R-lml_L)/(2*h)
                e[i] = 0
            if verbose:
                print '%s:'%key
                print abs((grad_an[key]-grad_num[key]))
                print ''

if 0:

    def predict(self,hyperparams,Fstar):
        """
        predict on Fstar
        """
        KV = self.get_covariances(hyperparams)
        Kstar = self.covar.K(hyperparams['covar'],self.F,Fstar)
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

        K = self.covar.K(hyperparams['covar'],self.F)
        
        if self.likelihood is not None:
            Knoise = self.likelihood.K(hyperparams['lik'],self.n)
            K += Knoise
        L = sp.linalg.cholesky(K).T# lower triangular

        alpha = sp.linalg.cho_solve((L,True),self.Y)
        Kinv = sp.linalg.cho_solve((L,True),sp.eye(L.shape[0]))
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


