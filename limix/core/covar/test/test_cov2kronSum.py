"""LMM testing code"""
import unittest
import scipy as sp
import numpy as np
import sys
from limix.core.covar import Cov2KronSum 
from limix.core.covar import FreeFormCov
from limix.utils.check_grad import mcheck_grad
from limix.utils.preprocess import covar_rescale

class TestCov2KronSum(unittest.TestCase):
    def setUp(self):
        sp.random.seed(1)
        # define row caoriance
        dim_r = 4
        X = sp.rand(dim_r, dim_r)
        R = covar_rescale(sp.dot(X,X.T))
        # define col covariances
        dim_c = 2
        Cg = FreeFormCov(dim_c)
        Cn = FreeFormCov(dim_c)
        self.C = Cov2KronSum(Cg = Cg, Cn = Cn, R = R)
        self.name = 'cov2kronSum'
        self.C.setRandomParams()

    def test_grad(self):
        def func(x, i):
            self.C.setParams(x)
            return self.C.K()

        def grad(x, i):
            self.C.setParams(x)
            return self.C.K_grad_i(i)

        x0 = self.C.getParams()
        err = mcheck_grad(func, grad, x0)

        np.testing.assert_almost_equal(err, 0., decimal = 6)

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

if __name__ == '__main__':
    unittest.main()
