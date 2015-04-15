import scipy as sp
from limix.core.utils.cached import *
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
        return sp.exp(self.params[0])

    #####################
    # Setters
    #####################
    @scale.setter
    def scale(self,value):
        assert value>=0, 'Scale must be >=0'
        self.params[0] = sp.log(value) 
        self.clear_all()
        self._notify()

    #####################
    # Params handling
    #####################
    def _calcNumberParams(self):
        self.n_params = 1

    #####################
    # Cached
    #####################
    @cached
    def K(self):
        return self.scale * self.K0

    #def Kcross(self):
    #    """
    #    evaluates the kernel between test and training points for given hyperparameters
    #    """
    #    return 0

    @cached
    def K_grad_i(self,i):
        if i==0:
            return self.scale*self.K0
        return None
