import sys
sys.path.insert(0,'./../../..')
from limix.core.utils.cached import *
import scipy as sp
import numpy as np
from covar_base import covariance
import pdb
import scipy.spatial as SS

class sqexp(covariance):
    """
    squared exponential covariance function
    """
    def __init__(self,X,Xstar=None):
        """
        X   dim x d input matrix
        """
        self.X = X
        self.Xstar = Xstar
        self._initParams()

    def get_input_dim(self):
        return self.X.shape[1]


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
    def scale_ste(self):
        if self.getFIinv() is None:
            R = None
        else:
            R = sp.sqrt(self.getFIinv()[0,0])
        return R

    @property
    def length_ste(self):
        if self.getFIinv() is None:
            R = None
        else:
            R = sp.sqrt(self.getFIinv()[1,1])
        return R

    @property
    def X(self):
        return self._X

    @property
    def Xstar(self):
        return self._Xstar

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
        self.initialize(value.shape[0])
        self.clear_all()
        self.clear_cache('E')
        self._notify()

    @Xstar.setter
    def Xstar(self,value):
        if value is None:
            self._use_to_predict = False
        else:
            assert value.shape[1]==self.X.shape[1], 'Dimension mismatch'
            self._use_to_predict = True
        self._Xstar = value
        self.clear_cache('Kcross')

    @covariance.use_to_predict.setter
    def use_to_predict(self,value):
        assert self.Xstar is not None, 'set Xstar!'
        self._use_to_predict = value

    #####################
    # Params handling
    #####################
    def _calcNumberParams(self):
        self.n_params = 2

    #####################
    # Cached
    #####################
    @cached
    def E(self):
        rv = SS.distance.pdist(self.X,'euclidean')**2
        rv = SS.distance.squareform(rv)
        return rv

    @cached
    def Kcross(self):
        assert self.Xstar.shape[1]==1, 'only implemented for 1-dim input'
        Estar = (self.Xstar - self.X.T)**2
        return  self.scale * sp.exp(-Estar/(2*self.length))

    @cached
    def K(self):
        return self.scale * sp.exp(-self.E()/(2*self.length))

    @cached
    def K_grad_i(self,i):
        r = self.K_grad_interParam_i(i)
        if i==0:
            r *= self.scale
        elif i==1:
            r *= self.length
        else:
            assert False, 'There is no index %d on sqexp.' %i
        return r

    ####################
    # Interpretable Params
    ####################
    def getInterParams(self):
        return SP.array([self.scale,self.length])

    def K_grad_interParam_i(self,i):
        if i==0:
            r = sp.exp(-self.E()/(2*self.length))
        else:
            A = sp.exp(-self.E()/(2*self.length))*self.E()
            r = self.scale * A / (2*self.length**2)
        return r
