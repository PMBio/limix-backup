import scipy as SP

class covariance:
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

    def getParams(self):
        """
        get hyperparameters
        """
        return self.params

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
        return None

    def _calcNumberParams(self):
        """
        calculates the number of parameters
        """
        pass

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
        
            

