"""LMM testing code"""
import unittest
import scipy as sp
import numpy as np
import sys
from limix.core.covar import Cov2KronSum 
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

    def test_solve(self):
        x = sp.randn(self.C.dim, 1)
        v1 = self.C.solve(x)
        v2 = Cov2KronSum.solve(self.C, x)
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

    def test_param_activation(self):
        self.C.act_Cr = False
        self.C.act_Cg = False
        self.C.act_Cn = False
        self.assertEqual(len(self.C.getParams()), 0)

        with self.assertRaises(ValueError):
            self.C.K_grad_i(0)

    def test_H_chol(self):
        C = self.C

        R  = sp.dot(C.W().T, C.DW())
        R += sp.eye(R.shape[0])
        chol1 = np.linalg.cholesky(R)
        chol2 = C.H_chol()

        np.testing.assert_array_almost_equal(chol1, chol2)

    def test_inv(self):
        C = self.C

        dL = C.d()[:,sp.newaxis] * C.L()
        WdL = sp.dot(C.DW().T, C.L())
        HiWdL = sp.dot(C.H_inv(), WdL)
        inv1 = sp.dot(C.L().T, dL) - sp.dot(WdL.T, HiWdL)
        inv2 = C.inv()

        np.testing.assert_array_almost_equal(inv1, inv2)

    def test_logdet(self):
        d1 = 2*sp.log(sp.diag(self.C.chol())).sum()
        d2 = self.C.logdet()
        np.testing.assert_almost_equal(d1, d2)

if __name__ == '__main__':
    unittest.main()
