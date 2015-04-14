import sys
sys.path.insert(0,'./../../..')
from limix.core.utils.cached import *
import scipy as sp
from covariance import covariance
import pdb
import scipy.spatial as SS

class sqexp(covariance):
    """
    squared exponential covariance function
    """
    def __init__(self,X):
        """
        X   dim x d input matrix
        """
        self.X = X

    #####################
    # Properties
    #####################
    @property
    def scale(self):
        return sp.exp(self.params[0])

    @property
    def length(self):
        return sp.exp(self.params[1])

    @property
    def X(self):
        return self._X

    #####################
    # Setters
    #####################
    @scale.setter
    def scale(self,value):
        assert value>=0, 'Scale must be >=0'
        self.params[0] = sp.log(value)
        self.clear_all()
        self._notify()

    @length.setter
    def length(self,value):
        assert value>=0, 'Length must be >=0'
        self.params[1] = sp.log(value)
        self.clear_all()
        self._notify()

    @X.setter
    def X(self,value):
        self._X = value
        covariance.__init__(self,self.X.shape[0])
        self.clear_all()
        self.clear_cache('E')
        self._notify()

    #####################
    # Params handling
    #####################
    def _calcNumberParams(self):
        self.n_params = self.X.shape[1]+1

    #####################
    # Cached
    #####################
    @cached
    def E(self):
        rv = SS.distance.pdist(self.X,'euclidean')**2
        rv = SS.distance.squareform(rv)
        return rv

    @cached
    def K(self):
        return self.scale * sp.exp(-self.E()/(2*self.length))

    #TODO
    #@cached
    #def Kcross(self):
    #    assert self.X.shape[1]==1, 'only implemented for 1-dim input'
    #    Estar = (self.Xstar - self.X.T)**2
    #    return  self.scale * sp.exp(-Estar/(2*self.length))

    @cached
    def K_grad_i(self,i):
        if i==0:
            return sp.exp(-self.E()/(2*self.length)) * self.scale
        else:
            A = sp.exp(-self.E()/(2*self.length))*self.E()
            return self.scale * A / (2*self.length)
