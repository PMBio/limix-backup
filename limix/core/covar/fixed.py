import scipy as sp
import numpy as np
import scipy.linalg as la
from hcache import cached
from limix.core.utils import assert_make_float_array
from limix.core.utils import assert_finite_array
from .covar_base import Covariance
import pdb

class FixedCov(Covariance):
    """
    Fixed-form covariance matrix.
    A fixed-form covariance matrix has only 1 parameter
    """
    def __init__(self, K0, Kcross0=None):
        """
        Args:
            K0:         semi-definite positive matrix that defines the fixed-form covariance
            Kcross0:    cross covariance between training and test samples
                        (used only for out-of-sample predictions)
        """
        Covariance.__init__(self)
        self._scale_act = True
        self.K0 = assert_make_float_array(K0, "K0")
        assert_finite_array(self.K0)

        if Kcross0 is not None:
            Kcross0 = assert_make_float_array(Kcross0, "Kcross0")
            assert_finite_array(Kcross0)

        self.Kcross0 = Kcross0
        self.scale = 1

    #####################
    # Properties
    #####################
    @property
    def scale(self):
        return self._scale 

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
    @cached('K0')
    def X0(self):
        S, U = la.eigh(self.K0)
        I = (S>1e-9)
        return U[:,I]*S[I]**(0.5) 

    @property
    @cached(['covar_base', 'K0'])
    def X(self):
        return sp.sqrt(self.scale) * self.X0

    @property
    def Kcross0(self):
        return self._Kcross0

    #####################
    # Setters
    #####################
    @scale.setter
    def scale(self,value):
        assert value >= 0, 'Scale must be >= 0.'
        self._scale = value 
        self.clear_all()

    @K0.setter
    def K0(self,value):
        self._K0 = value
        self.initialize(value.shape[0])
        self.clear_cache('K0')
        self._notify()

    @Kcross0.setter
    def Kcross0(self,value):
        if value is None:
            self._use_to_predict = False
        else:
            assert value.shape[1] == self.dim, 'Dimension mismatch.'
            self._use_to_predict = True
        self._Kcross0 = value
        self.clear_cache('Kcross0')
        self._notify()

    @Covariance.use_to_predict.setter
    def use_to_predict(self,value):
        assert self.Kcross0 is not None, 'Kcross0 has to be set before.'
        self._use_to_predict = value

    #####################
    # Activation handling
    #####################
    @property
    def act_scale(self):
        return self._scale_act

    @act_scale.setter
    def act_scale(self, act):
        self._scale_act = bool(act)
        self._notify()

    #####################
    # Params handling
    #####################
    def setParams(self, params):
        if int(self._scale_act) != len(params):
            raise ValueError("The number of parameters passed to setParams "
                             "differs from the number of active parameters.")
        if self._scale_act:
            self.scale = sp.exp(params[0])
            self.clear_all()

    def _calcNumberParams(self):
        self.n_params = 1

    def getParams(self):
        if self._scale_act:
            return sp.array([sp.log(self.scale)])
        return np.array([])

    def getNumberParams(self):
        """
        return the number of hyperparameters
        """
        return int(self._scale_act)

    #####################
    # Cached
    #####################
    @cached(['covar_base', 'K0'])
    def K(self):
        return self.scale * self.K0

    @cached(['Kcross0', 'covar_base'])
    def Kcross(self):
        return self.scale * self.Kcross0

    @cached(['covar_base', 'K0'])
    def K_grad_i(self,i):
        if i >= int(self._scale_act):
            raise ValueError("Trying to retrieve the gradient over a "
                             "parameter that is inactive.")
        r = self.scale * self.K0
        return r

    @cached(['covar_base', 'K0'])
    def K_hess_i_j(self, i, j):
        if i >= int(self._scale_act) or j >= int(self._scale_act):
            raise ValueError("Trying to retrieve the hessian over a "
                             "parameter that is inactive.")
        return self.K()

    @cached(['covar_base', 'K0'])
    def Xgrad(self, i):
        return 0.5*self.X

    ####################
    # Interpretable Params
    ####################
    def getInterParams(self):
        if self._scale_act:
            return SP.array([self.scale])
        return np.array([])

    def K_grad_interParam_i(self,i):
        if i >= int(self._scale_act):
            raise ValueError("Trying to retrieve the gradient over a "
                             "parameter that is inactive.")

        return self.K0


if __name__=='__main__':

    import pdb
    C = FixedCov(sp.ones((2,2)))
    pdb.set_trace()

