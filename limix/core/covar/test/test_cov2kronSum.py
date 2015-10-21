"""LMM testing code"""
import unittest
import scipy as sp
import numpy as np
import sys
from limix.core.covar import Covariance
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

    def test_solve(self):
        x = sp.randn(self.C.dim, 1)
        v1 = self.C.solve(x)
        v2 = Covariance.solve(self.C, x)
        np.testing.assert_almost_equal(v1, v2, decimal = 5)

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

    def test_inv(self):
        inv1 = np.dot(self.C.L().T, self.C.d()[:, sp.newaxis] * self.C.L())
        inv2 = self.C.inv()
        np.testing.assert_array_almost_equal(inv1, inv2)

    def test_logdet(self):
        d1 = 2*np.log(np.diag(self.C.chol())).sum()
        d2 = self.C.logdet()
        np.testing.assert_almost_equal(d1, d2)


if __name__ == '__main__':
    unittest.main()
