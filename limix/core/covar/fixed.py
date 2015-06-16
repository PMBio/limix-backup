import scipy as sp
import numpy as np
from limix.core.type.cached import cached
from limix.core.utils import assert_make_float_array
from limix.core.utils import assert_finite_array
from covar_base import Covariance
import pdb

class FixedCov(Covariance):
    """
    squared exponential covariance function
    """
    def __init__(self, K0, Kcross0=None):
        Covariance.__init__(self)
        self.K0 = assert_make_float_array(K0, "K0")
        assert_finite_array(self.K0)

        if Kcross0 is not None:
            Kcross0 = assert_make_float_array(Kcross0, "Kcross0")
            assert_finite_array(Kcross0)

        self.Kcross0 = Kcross0

    #####################
    # Properties
    #####################
    @property
    def scale(self):
        return sp.exp(self.params[0])

    @property
    def scale_ste(self):
        if self.getFIinv() is None:
            R = None
        else:
            R = sp.sqrt(self.getFIinv()[0,0])
        return R

    @property
    def K0(self):
        return self._K0

    @property
    def Kcross0(self):
        return self._Kcross0

    #####################
    # Setters
    #####################
    @scale.setter
    def scale(self,value):
        assert value >= 0, 'Scale must be >= 0.'
        self.params[0] = sp.log(value)
        self.clear_all()
        self._notify()

    @K0.setter
    def K0(self,value):
        self._K0 = value
        self.initialize(value.shape[0])
        self._notify()

    @Kcross0.setter
    def Kcross0(self,value):
        if value is None:
            self._use_to_predict = False
        else:
            assert value.shape[1] == self.dim, 'Dimension mismatch.'
            self._use_to_predict = True
        self._Kcross0 = value
        self.clear_cache('Kcross')
        self._notify()

    @Covariance.use_to_predict.setter
    def use_to_predict(self,value):
        assert self.Kcross0 is not None, 'Kcross0 has to be set before.'
        self._use_to_predict = value

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

    @cached
    def Kcross(self):
        return self.scale * self.Kcross0

    @cached
    def K_grad_i(self,i):
        r = self.scale * self.K0
        return r

    ####################
    # Interpretable Params
    ####################
    def getInterParams(self):
        return SP.array([self.scale])

    def K_grad_interParam_i(self,i):
        return self.K0
