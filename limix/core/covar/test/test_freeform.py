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

        np.testing.assert_almost_equal(err, 0., decimal=6)

    def test_param_activation(self):
        self.assertEqual(len(self.C.getParams()), 10)
        self.C.act_K = False
        self.assertEqual(len(self.C.getParams()), 0)

        self.C.setParams(np.array([]))
        with self.assertRaises(ValueError):
            self.C.setParams(np.array([0]))

        with self.assertRaises(ValueError):
            self.C.K_grad_i(0)

    def test_Khess(self):

        cov = self.C

        for j in range(cov.getNumberParams()):

            def func(x, i):
                cov.setParams(x)
                return cov.K_grad_i(j)

            def grad(x, i):
                cov.setParams(x)
                return cov.K_hess_i_j(j, i)

            x0 = cov.getParams()
            err = mcheck_grad(func, grad, x0)
            np.testing.assert_almost_equal(err, 0.)

if __name__ == '__main__':
    unittest.main()
