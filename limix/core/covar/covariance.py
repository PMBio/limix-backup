import sys
sys.path.insert(0,'./../../..')
from limix.core.cobj import *
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
        self.clear_cache('inv','chol','S','U','Sgrad','Ugrad')

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
    def S(self):
        RV,U = LA.eigh(self.K()) 
        self.fill_cache('U',U)
        return RV

    @cached
    def U(self):
        S,RV = LA.eigh(self.K()) 
        self.fill_cache('S',S)
        return RV

    def Sgrad(self,i):
        Kgrad = self.Kgrad_param(i)
        RV = (self.U()*SP.dot(Kgrad,self.U())).sum(0)
        return RV

    def Ugrad(self,i):
        """
        dv_p = (lambda_p I - A)^(+)*dA*U_p
        dv_p = (lambda_p I - USUt)^(+)*dA*U_p
        dv_p = (U * (lambda_p I - S) Ut)^(+) *dA*U_p
        """
        Kgrad = self.Kgrad_param(i)
        RV = SP.zeros((self.P,self.P))
        for p in range(self.P):
            S1 = self.S()[p]-self.S()
            I  = abs(S1)>1e-4
            _U = self.U()[:,I]
            _D = (S1[I]**(-1))[:,SP.newaxis]
            pseudo = SP.dot(_U,_D*_U.T) 
            RV[:,p] = SP.dot(pseudo,SP.dot(Kgrad,self.U()[:,p]))
        return RV


