import pdb
import scipy as sp
import scipy.linalg
import copy
import sys
sys.path.insert(0,'./../../..')
from limix.core.utils.observed import Observed
from limix.core.utils.cached import *
import limix.core.mean.mean_base
from limix.core.covar.cov_reml import cov_reml

class gp(cObject, Observed):
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
        self.Areml = cov_reml(self)
        self.clear_all()
        self.update_B()

    def clear_all(self):
        self.clear_Areml()
        self.clear_lml_terms()
        self.clear_lmlgrad_terms_i()
        self.clear_lmlgrad_terms()

    def clear_Areml(self):
        self._notify()

    def clear_lml_terms(self):
        self.clear_cache('KiF','YKiF','KiFB','KiY','YKiY','YKiFB','LML')

    def clear_lmlgrad_terms_i(self):
        self.clear_cache('DiKKiY','DiKKiF','DiKKiFB',
                            'YKiY_grad_i','YKiFB_grad_i')

    def clear_lmlgrad_terms(self):
        self.clear_cache('YKiY_grad','YKiFB_grad','Areml_logdet_grad','LML_grad')

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

    ######################
    # Areml
    ######################
    def Areml_K(self):
        return sp.dot(self.mean.F.T,self.KiF())

    def Areml_K_grad_i(self,i):
        return -sp.dot(self.KiF().T,self.DiKKiF(i))

    #######################
    # LML terms
    #######################
    @cached
    def KiF(self):
        return self.covar.solve(self.mean.F)

    @cached
    def YKiF(self):
        return sp.dot(self.mean.Y.T,self.KiF())

    # B is calculated here but cached in the mean?
    def update_B(self):
        self.mean.B = self.Areml.solve(self.YKiF().T)

    @cached
    def KiFB(self):
        return sp.dot(self.KiF(),self.mean.B)

    @cached
    def KiY(self):
        return self.covar.solve(self.mean.Y)

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
    def DiKKiY(self,i):
        return sp.dot(self.covar.K_grad_i(i),self.KiY())

    @cached
    def DiKKiF(self,i):
        return sp.dot(self.covar.K_grad_i(i),self.KiF())

    @cached
    def DiKKiFB(self,i):
        return sp.dot(self.DiKKiF(i),self.mean.B)


    @cached
    def YKiY_grad_i(self,i):
        return -(self.KiY()*self.DiKKiY(i)).sum()

    @cached
    def YKiFB_grad_i(self,i):
        rv = -2*(self.KiY()*self.DiKKiFB(i)).sum()
        rv+= (self.KiFB()*self.DiKKiFB(i)).sum()
        return rv

    #######################
    # LML and gradients
    #######################

    @cached
    def LML(self):
        #const term to add?
        rv = -0.5*self.covar.logdet()
        rv -= 0.5*self.Areml.logdet()
        rv -= 0.5*self.YKiY()
        rv += 0.5*self.YKiFB()
        return rv

    @cached
    def YKiY_grad(self):
        n_params = self.getParams()['covar'].shape[0]
        RV = {'covar': sp.zeros(n_params)}
        for i in range(n_params):
            RV['covar'][i] = self.YKiY_grad_i(i)
        return RV

    @cached
    def YKiFB_grad(self):
        n_params = self.getParams()['covar'].shape[0]
        RV = {'covar': sp.zeros(n_params)}
        for i in range(n_params):
            RV['covar'][i] = self.YKiFB_grad_i(i)
        return RV

    @cached
    def Areml_logdet_grad(self):
        n_params = self.getParams()['covar'].shape[0]
        RV = {'covar': sp.zeros(n_params)}
        for i in range(n_params):
            RV['covar'][i] = self.Areml.logdet_grad_i(i)
        return RV

    @cached
    def LML_grad(self):
        n_params = self.getParams()['covar'].shape[0]
        RV = {'covar': sp.zeros(n_params)}
        for i in range(n_params):
            RV['covar'][i] = -0.5*self.covar.logdet_grad_i(i)
            RV['covar'][i] -= 0.5*self.Areml.logdet_grad_i(i)
            RV['covar'][i] -= 0.5*self.YKiY_grad_i(i)
            RV['covar'][i] += 0.5*self.YKiFB_grad_i(i)
        return RV

    def predict(self):
        R = None
        if self.covar.use_to_predict:
            Kcross = self.covar.Kcross()
            KiYres = self.KiY()-self.KiFB()
            R = SP.dot(Kcross,KiYres)
        if self.mean.use_to_predict:
            _ = self.mean.predict()
            if R is None:
                R = _ 
            else:
                assert _.shape[0]==R.shape[0], 'Dimension mismatch'
                assert _.shape[1]==R.shape[1], 'Dimension mismatch'
                R += _
            return R

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

