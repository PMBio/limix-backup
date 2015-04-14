import scipy as sp
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
        self._notify()

    #####################
    # Params handling
    #####################
    def setParams(self,params):
        self.scale  = sp.exp(params[0])

    def getParams(self):
        params = sp.log(sp.array([self.scale]))
        return params

    def _calcNumberParams(self):
        self.n_params = 1

    #####################
    # Cached
    #####################
    def K(self):
        return self.scale * self.K0

    #def Kcross(self):
    #    """
    #    evaluates the kernel between test and training points for given hyperparameters
    #    """
    #    return 0

    def K_grad_i(self):
        if self._grad_idx==0:
            return self.scale*self.K0
        return None
