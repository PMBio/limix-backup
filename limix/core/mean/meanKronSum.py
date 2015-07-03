import sys
import scipy as sp
import numpy as np
import scipy.linalg as LA
import copy
import pdb

from mean_base import MeanBase
from limix.utils.preprocess import regressOut
from limix.utils.util_functions import to_list
from limix.core.type.cached import *
from limix.core.type.observed import *
from limix.core.utils import assert_make_float_array
from limix.core.utils import assert_type_or_list_type


class MeanKronSum(MeanBase):
    """
    Sum of Kronecker Mean for multi-trait gp regression
    Notation:
        N = number of individuals
        No = number of out-of-sample individuals for predictions
        P = number of traits
    """

    def __init__(self, Y, F=None, A=None, Fstar=None):
        """
        Args:
            Y:        phenotype matrix [N, P]
            F:        list of sample fixed effect designs.
                      Each term must have first dimension N
            A:        list of trait fixed effect design.
                      Each term must have second dimension P
            Fstar:    list out-of-sample fixed effect design.
                      Each term must have first dimension No
        """
        Cached.__init__(self)
        Y = assert_make_float_array(Y, 'Y')
        if F is not None:
            try:
                assert_type_or_list_type(F, np.ndarray, 'F')
            except TypeError as e:
                raise TypeError(e.message + ' Parameter F might also be set'
                                ' to None.')

        if A is not None:
            try:
                assert_type_or_list_type(A, np.ndarray, 'A')
            except TypeError as e:
                raise TypeError(e.message + ' Parameter A might also be set'
                                ' to None.')

        assert Fstar is None, 'This constructor still does not support Fstar.'

        self.Y = Y
        self.setDesigns(F, A)
        self.Fstar = Fstar
        self.setFIinv(None)

    #########################################
    # Properties
    #########################################
    @property
    def n_terms(self):
        return self._n_terms

    @property
    def n_covs(self):
        return self._n_covs

    @property
    def Y(self):
        return self._Y

    @property
    def F(self):
        return self._F

    @property
    def A(self):
        return self._A

    @property
    def B(self):
        B = []
        istart = 0
        for ti in range(self.n_terms):
            iend = istart + self.F[ti].shape[1] * self.A[ti].shape[0]
            B.append(sp.reshape(self.b[istart:iend], (self.F[ti].shape[1],
                     self.A[ti].shape[0]), order='F'))
            istart = iend
        return B

    @property
    def B_ste(self):
        print 'TODO: implement me'

    @property
    def y(self):
        return sp.reshape(self.Y, (self._N*self._P, 1), order='F')

    @property
    def b(self):
        return self._b

    @property
    def b_ste(self):
        if self.getFIinv() is None:
            R = None
        else:
            R = sp.sqrt(self.getFIinv().diagonal())[:, sp.newaxis]
        return R

    @property
    def Fstar(self):
        print 'TODO: assert stuff'
        return self._Fstar

    @property
    def use_to_predict(self):
        return self._use_to_predict

    @property
    @cached
    def W(self):
        R = sp.zeros((self.Y.size, self.n_covs))
        istart = 0
        for ti in range(self.n_terms):
            iend = istart + self.F[ti].shape[1] * self.A[ti].shape[0]
            R[:, istart:iend] = sp.kron(self.A[ti].T, self.F[ti])
        return R

    #########################################
    # Utils function
    #########################################
    def Ft_dot(self, M):
        dim0 = 0
        R = sp.zeros((self._k, M.shape[1]))
        istart = 0
        for ti in range(self.n_terms):
            _dim = self.F[ti].shape[1]
            iend = istart + _dim
            R[istart:iend] = sp.dot(self.F[ti].T, M)
            istart = iend
        return R

    #########################################
    # Setters
    #########################################
    @Y.setter
    def Y(self, value):
        """ set phenotype """
        self._N = value.shape[0]
        self._P = value.shape[1]
        self._Y = value
        self._notify()
        self._notify('pheno')

    def setDesigns(self, F, A):
        """ set fixed effect designs """
        F = to_list(F)
        A = to_list(A)
        assert len(A) == len(F), 'MeanKronSum: A and F must have same length!'
        n_terms = len(F)
        n_covs = 0
        k = 0
        l = 0
        for ti in range(n_terms):
            assert F[ti].shape[0] == self._N, 'MeanKronSum: Dimension mismatch'
            assert A[ti].shape[1] == self._P, 'MeanKronSum: Dimension mismatch'
            n_covs += F[ti].shape[1] * A[ti].shape[0]
            k += F[ti].shape[1]
            l += A[ti].shape[0]
        self._n_terms = n_terms
        self._n_covs = n_covs
        self._k = k
        self._l = l
        self._F = F
        self._A = A
        self._b = sp.zeros((n_covs, 1))
        self.clear_cache('predict_in_sample', 'Yres', 'W')
        self._notify('designs')
        self._notify()

    @Fstar.setter
    def Fstar(self, value):
        """ set fixed effect design for predictions """
        if value is None:
            self._use_to_predict = False
        else:
            assert value.shape[1] == self._K, 'Dimension mismatch'
            self._use_to_predict = True
        self._Fstar = value
        self.clear_cache('predict')

    @b.setter
    def b(self, value):
        assert value.shape[0] == self._n_covs, 'Dimension mismatch'
        assert value.shape[1] == 1, 'Dimension mismatch'
        self._b = value
        self.clear_cache('predict_in_sample', 'Yres', 'predict')

    @use_to_predict.setter
    def use_to_predict(self, value):
        assert not (self.Fstar is None and value is True), 'set Fstar!'
        self._use_to_predict = value

    #########################################
    # Predictions
    #########################################
    @cached
    def predict(self):
        r = self._predict_fun(self.Fstar)
        return r

    @cached
    def predict_in_sample(self):
        r = self._predict_fun(self.F)
        return r

    def _predict_fun(self, M):
        assert len(M) == self.n_terms, 'MeanKronSum: Dimension mismatch'
        rv = sp.zeros((self._N, self._P))
        for ti in range(self.n_terms):
            rv += sp.dot(sp.dot(M[ti], self.B[ti]), self.A[ti])
        return rv

    @cached
    def Yres(self):
        """ residual """
        RV = self.Y - self.predict_in_sample()
        return RV

    #######################################
    # Standard errors
    ########################################
    def setFIinv(self, value):
        self._FIinv = value

    def getFIinv(self):
        return self._FIinv

if __name__ == '__main__':

    # define phenotype
    N = 1000
    P = 4
    Y = sp.randn(N, P)

    # define fixed effects
    F = []
    A = []
    F.append(sp.randn(N, 3))
    F.append(sp.randn(N, 2))
    A.append(sp.eye(P))
    A.append(sp.ones((1, P)))

    pdb.set_trace()

    mean = MeanKronSum(Y, F, A)
