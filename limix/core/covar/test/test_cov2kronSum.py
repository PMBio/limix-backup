"""LMM testing code"""
import unittest
import scipy as sp
import numpy as np
import sys
from limix.core.covar import Cov2KronSum
from limix.core.covar import FreeFormCov
from limix.utils.check_grad import mcheck_grad
from limix.utils.preprocess import covar_rescale
from limix.core.type.exception import TooExpensiveOperationError

class TestCov2KronSum(unittest.TestCase):
    def setUp(self):
        sp.random.seed(1)
        # define row caoriance
        dim_r = 4
        X = sp.rand(dim_r, dim_r)
        self.R = covar_rescale(sp.dot(X,X.T))
        # define col covariances
        dim_c = 2
        Cg = FreeFormCov(dim_c)
        Cn = FreeFormCov(dim_c)
        self.C = Cov2KronSum(Cg = Cg, Cn = Cn, R = self.R)
        self.name = 'cov2kronSum'
        self.C.setRandomParams()

    def test_logdet_grad(self):
        def func(x, i):
            self.C.setParams(x)
            return self.C.logdet()

        def grad(x, i):
            self.C.setParams(x)
            return self.C.logdet_grad_i(i)

        x0 = self.C.getParams()
        err = mcheck_grad(func, grad, x0)
        np.testing.assert_almost_equal(err, 0., decimal = 5)

    def test_too_expensive_exceptions(self):
        Cg = FreeFormCov(5001)
        Cn = FreeFormCov(5001)
        C = Cov2KronSum(Cg=Cg, Cn=Cn, R=self.R)
        with self.assertRaises(TooExpensiveOperationError):
            C.L()
        with self.assertRaises(TooExpensiveOperationError):
            C.K()
        with self.assertRaises(TooExpensiveOperationError):
            C.K_grad_i(0)

    def test_param_activation(self):
        self.C.act_Cg = False
        self.C.act_Cn = False
        self.assertEqual(len(self.C.getParams()), 0)

        with self.assertRaises(ValueError):
            self.C.K_grad_i(0)

    # def test_inv(self):
    #
    #     def inv_debug(self):
    #         return sp.dot(self.L().T, self.d()[:, sp.newaxis] * self.L())


if __name__ == '__main__':
    unittest.main()
