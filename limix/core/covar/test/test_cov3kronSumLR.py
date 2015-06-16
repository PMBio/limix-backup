"""LMM testing code"""
import unittest
import scipy as sp
import numpy as np
import sys
from limix.core.covar import Cov3KronSumLR
from limix.core.covar import FreeFormCov
from limix.utils.check_grad import mcheck_grad
from limix.utils.preprocess import covar_rescale
from limix.core.type.exception import TooExpensiveOperationError

class TestCov3KronSumLR(unittest.TestCase):
    def setUp(self):
        sp.random.seed(1)
        # define row caoriance
        dim_r = 4
        rank_r = 2
        G = sp.rand(dim_r, rank_r)
        X = sp.rand(dim_r, dim_r)
        R = covar_rescale(sp.dot(X,X.T))
        # define col covariances
        dim_c = 2
        Cg = FreeFormCov(dim_c)
        Cn = FreeFormCov(dim_c)
        self.C = Cov3KronSumLR(Cn = Cn, Cg = Cg, R = R, G = G, rank = 1)
        self.name = 'cov3kronSumLR'
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
        dim_r = 100
        rank_r = 2
        G = sp.rand(dim_r, rank_r)
        X = sp.rand(dim_r, dim_r)
        R = covar_rescale(sp.dot(X,X.T))

        dim_c = 5001
        Cg = FreeFormCov(dim_c)
        Cn = FreeFormCov(dim_c)
        C = Cov3KronSumLR(Cn = Cn, Cg = Cg, R = R, G = G, rank = 1)

        with self.assertRaises(TooExpensiveOperationError):
            C.L()
        with self.assertRaises(TooExpensiveOperationError):
            C.K()
        with self.assertRaises(TooExpensiveOperationError):
            C.K_grad_i(0)

if __name__ == '__main__':
    unittest.main()
