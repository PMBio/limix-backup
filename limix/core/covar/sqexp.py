import sys
sys.path.insert(0,'./../../..')
from limix.core.cobj import *
import scipy as SP
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
        return self._scale

    @property
    def length(self):
        return self._length

    @property
    def X(self):
        return self._X

    #####################
    # Setters
    #####################
    @scale.setter
    def scale(self,value):
        assert value>=0, 'Scale must be >=0'
        self._scale = value
        self.clear_all()
        self._notify()

    @length.setter
    def length(self,value):
        assert value>=0, 'Length must be >=0'
        self._length = value
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
    def setParams(self,params):
        self.scale  = SP.exp(params[0])
        self.length = SP.exp(params[1])

    def getParams(self):
        params = SP.log(SP.array([self.scale,self.length]))
        return params

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
        return self.scale * SP.exp(-self.E()/(2*self.length))

    #TODO
    #@cached
    #def Kcross(self):
    #    assert self.X.shape[1]==1, 'only implemented for 1-dim input'
    #    Estar = (self.Xstar - self.X.T)**2
    #    return  self.scale * SP.exp(-Estar/(2*self.length))

    @cached
    def K_grad_i(self):
        if self._grad_idx==0:
            return SP.exp(-self.E()/(2*self.length)) * self.scale
        else:
            A = SP.exp(-self.E()/(2*self.length))*self.E()
            return self.scale * A / (2*self.length)
