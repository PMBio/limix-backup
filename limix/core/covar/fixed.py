import scipy as SP
from covariance import covariance
import pdb

class fixed(covariance):
    """
    squared exponential covariance function
    """
    def __init__(self,K):
        """
        initialization
        """
        covariance.__init__(self,K.shape[0])
        self.K0 = K

    #####################
    # Properties
    #####################
    @property
    def scale(self):
        return self._scale

    #####################
    # Setters
    #####################
    @scale.setter
    def scale(self,value):
        assert value>=0, 'Scale must be >=0'
        self._scale = value
        self.clear_all()

    #####################
    # Params handling
    #####################
    def setParams(self,params):
        self.scale  = SP.exp(params[0])

    def getParams(self,params):
        params = SP.log(SP.array([self.scale]))

    def _calcNumberParams(self):
        self.n_params = 1

    #####################
    # Cached
    #####################
    def K(self):
        return self.scale * self.cov
        
    #def Kcross(self):
    #    """
    #    evaluates the kernel between test and training points for given hyperparameters
    #    """
    #    return 0
    
    def Kgrad_param_i(self):
        if grad_idx==0:
            return self.scale*self.cov
        return None




    
