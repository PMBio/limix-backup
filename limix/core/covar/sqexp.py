import sys
from hcache import cached
from limix.core.utils import assert_make_float_array
from limix.core.utils import assert_finite_array
import scipy as sp
import numpy as np
from .covar_base import Covariance
import scipy.spatial as SS

class SQExpCov(Covariance):
    """
    Unidimensional squared exponential covariance function (kernel) for GP regression.
    A unidimensional squared exponential covariance function has two parameters:
        scale:      scale of the covariance (propto explained variance)
        length:     length scale of the input dimension
    """
    def __init__(self, X, Xstar=None):
        """
        X:          [dim, 1] input matrix
        Xstar:      [dim_star, 1] out-of-sample input matrix
        """
        Covariance.__init__(self)
        self._scale_act = True
        self._length_act = True

        X = assert_make_float_array(X, "X")
        assert_finite_array(X)
        self.X = X

        if Xstar is not None:
            Xstar = assert_make_float_array(Xstar, "Xstar")
            assert_finite_array(Xstar)

        self.Xstar = Xstar
        self.params = np.zeros(2)

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
        if not self._scale_act:
            raise ValueError("Scale is an inactive.")
        if self.getFIinv() is None:
            R = None
        else:
            i = self._index2actindex(0)
            R = sp.sqrt(self.getFIinv()[i,i])
        return R

    @property
    def length_ste(self):
        if not self._length_act:
            raise ValueError("Length is an inactive.")
        if self.getFIinv() is None:
            R = None
        else:
            i = self._index2actindex(1)
            R = sp.sqrt(self.getFIinv()[i,i])
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

    @length.setter
    def length(self,value):
        assert value>=0, 'Length must be >=0'
        self.params[1] = sp.log(value)
        self.clear_all()

    @X.setter
    def X(self,value):
        self._X = value
        self.initialize(value.shape[0])
        self.clear_cache('X')
        self.clear_all()

    @Xstar.setter
    def Xstar(self,value):
        if value is None:
            self._use_to_predict = False
        else:
            assert value.shape[1]==self.X.shape[1], 'Dimension mismatch'
            self._use_to_predict = True
        self._Xstar = value
        self.clear_cache('Xstar')
        self._notify()

    @Covariance.use_to_predict.setter
    def use_to_predict(self,value):
        assert self.Xstar is not None, 'set Xstar!'
        self._use_to_predict = value
        self._notify()

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

    @property
    def act_length(self):
        return self._length_act

    @act_length.setter
    def act_length(self, act):
        self._length_act = bool(act)
        self._notify()

    def _actindex2index(self, i):
        return i + int(not self._scale_act)

    def _index2actindex(self, i):
        return i - int(not self._scale_act)

    #####################
    # Params handling
    #####################
    def setParams(self, params):
        sel = np.asarray((self._scale_act, self._length_act))
        if np.sum(sel) != len(params):
            raise ValueError("The number of parameters passed to setParams "
                             "differs from the number of active parameters.")
        self.params[sel] = params
        self.clear_all()

    def getParams(self):
        sel = np.asarray((self._scale_act, self._length_act))
        return self.params[sel]

    def getNumberParams(self):
        return np.sum([self._scale_act, self._length_act])

    #####################
    # Cached
    #####################
    @cached('X')
    def E(self):
        rv = SS.distance.pdist(self.X,'euclidean')**2
        rv = SS.distance.squareform(rv)
        return rv

    @cached(['X', 'Xstar', 'covar_base'])
    def Kcross(self):
        assert self.Xstar.shape[1]==1, 'only implemented for 1-dim input'
        Estar = (self.Xstar - self.X.T)**2
        return  self.scale * sp.exp(-Estar/(2*self.length))

    @cached('covar_base')
    def K(self):
        return self.scale * sp.exp(-self.E()/(2*self.length))

    @cached('covar_base')
    def K_grad_i(self, i):
        if i >= int(self._scale_act) + int(self._length_act):
            raise ValueError("Trying to retrieve the gradient over a "
                             "parameter that is inactive.")

        r = self.K_grad_interParam_i(i)
        i = self._actindex2index(i)
        if i==0:
            r *= self.scale
        elif i==1:
            r *= self.length
        else:
            assert False, 'There is no index %d on sqexp.' %i
        return r

    @cached('covar_base')
    def K_hess_i_j(self, i, j):
        if i >= int(self._scale_act) + int(self._length_act) or j >= int(self._scale_act) + int(self._length_act):
            raise ValueError("Trying to retrieve the hessian over a "
                             "parameter that is inactive.")

        i = self._actindex2index(i)
        j = self._actindex2index(j)
        if i==0 and j==0:
            r = self.K()
        elif (i==0 and j==1) or (i==1 and j==0):
            r = self.K_grad_i(1)
        elif i==1 and j==1:
            r = self.K_grad_i(1) - self.K_grad_i(0)
            r*= self.E() / (2*self.length)
        else:
            assert False, 'There is no index %d on sqexp.' %i
        return r

    ####################
    # Interpretable Params
    ####################
    def getInterParams(self):
        params = []
        if self._scale_act:
            params.append(self.scale)
        if self._length_act:
            params.append(self.length)
        return np.array(params)

    def K_grad_interParam_i(self, i):
        i = self._actindex2index(i)
        if i == 0:
            r = sp.exp(-self.E()/(2*self.length))
        elif i == 1:
            A = sp.exp(-self.E()/(2*self.length))*self.E()
            r = self.scale * A / (2*self.length**2)
        else:
            raise ValueError("Trying to retrieve the gradient over a "
                             "parameter that is inactive.")
        return r
