"""LMM testing code"""
import unittest
import scipy as SP
import numpy as np
import sys
from limix.core.covar import FreeFormCov
from limix.utils.check_grad import mcheck_grad

class TestFreeForm(unittest.TestCase):
    def setUp(self):
        SP.random.seed(1)
        self.n=4
        self.C = FreeFormCov(self.n)
        self.name = 'freeform'
        self.n_params=self.C.getNumberParams()
        params=SP.randn(self.n_params)
        self.C.setParams(params)

    def test_grad(self):
        def func(x, i):
            self.C.setParams(x)
            return self.C.K()

        def grad(x, i):
            self.C.setParams(x)
            return self.C.K_grad_i(i)

        x0 = self.C.getParams()
        err = mcheck_grad(func, grad, x0)

        np.testing.assert_almost_equal(err, 0.)

if __name__ == '__main__':
    unittest.main()
