import sys
sys.path.insert(0,'./../../..')
from limix.core.cobj import *
from limix.core.utils.eigen import *
import scipy as SP
import scipy.linalg as LA
import warnings

class covariance(cObject):
    """
    abstract super class for all implementations of covariance functions
    """
    def __init__(self,P):
        self.P = P
        self._calcNumberParams()
        self._initParams()

    def setParams(self,params):
        """
        set hyperparameters
        """
        self.params = params
        self.params_have_changed=True
        self.clear_cache('inv','chol','S','U','USi2','Sgrad','Ugrad')

    def setRandomParams(self):
        """
        set random hyperparameters
        """
        params = SP.randn(self.getNumberParams())
        self.setParams(params)

    def setCovariance(self,cov):
        """
        set hyperparameters from given covariance
        """
        warnings.warn('not implemented')

    def getParams(self):
        """
        get hyperparameters
        """
        return self.params

    def perturbParams(self,pertSize=1e-3):
        """
        slightly perturbs the values of the parameters
        """
        params = self.getParams()
        self.setParams(params+pertSize*SP.randn(params.shape[0]))

    def getNumberParams(self):
        """
        return the number of hyperparameters
        """
        return self.n_params

    def K(self):
        """
        evaluates the kernel for given hyperparameters theta and inputs X
        """
        LG.critical("implement K")
        print("%s: Function K not yet implemented"%(self.__class__))
        return None
     
    def Kgrad_param(self,i):
        """
        partial derivative with repspect to the i-th hyperparamter theta[i]
        """
        LG.critical("implement Kgrad_theta")
        print("%s: Function K not yet implemented"%(self.__class__))

    def _calcNumberParams(self):
        """
        calculates the number of parameters
        """
        warnings.warn('not implemented')

    def _initParams(self):
        """
        initialize paramters to vector of zeros
        """
        params = SP.zeros(self.n_params)
        self.setParams(params)

    def Kgrad_param_num(self,i,h=1e-4):
        """
        check discrepancies between numerical and analytical gradients
        """
        params  = self.params.copy()
        e = SP.zeros_like(params); e[i] = 1
        self.setParams(params-h*e)
        C_L = self.K()
        self.setParams(params+h*e)
        C_R = self.K()
        self.setParams(params)
        RV = (C_R-C_L)/(2*h)
        return RV

    @cached
    def chol(self):
        return LA.cholesky(self.K()).T
            
    @cached
    def inv(self):
        return LA.cho_solve((self.chol(),True),SP.eye(self.P)) 

    @cached
    def logdet(self):
        return 2*SP.log(SP.diag(self.chol())).sum()

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

    def Sgrad(self,i):
        return dS_dti(self.Kgrad_param(i),U=self.U())

    def Ugrad(self,i):
        return dU_dti(self.Kgrad_param(i),U=self.U(),S=self.S())

    def USi2grad(self,i):
        # dU * S**(-1/2) + U * (-1/2 S**(-3/2) dS)
        Si2grad = -0.5*self.S()**(-1.5)*self.Sgrad(i)
        return self.Ugrad(i)*(self.S()**(-0.5)) + self.U()*Si2grad

