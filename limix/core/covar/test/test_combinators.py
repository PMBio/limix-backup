import unittest
import numpy as np
from limix.core.covar import SQExpCov
from limix.core.covar import SumCov
from limix.utils.check_grad import mcheck_grad
import scipy as sp

class TestSumCov(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)
        self._X1 = np.random.randn(10, 5)
        self._X2 = np.random.randn(10, 8)
        self._cov1 = SQExpCov(self._X1)
        self._cov2 = SQExpCov(self._X2)
        self._cov = SumCov(self._cov1, self._cov2)

    def test_sum_combination(self):
        K1 = self._cov1.K() + self._cov2.K()
        K2 = self._cov.K()

        np.testing.assert_almost_equal(K1, K2)

    def test_Kgrad(self):

        cov = self._cov

        def func(x, i):
            cov.setParams(x)
            return cov.K()

        def grad(x, i):
            cov.setParams(x)
            return cov.K_grad_i(i)

        x0 = cov.getParams()
        err = mcheck_grad(func, grad, x0)

        np.testing.assert_almost_equal(err, 0.)

    def test_Khess(self):

        cov = self._cov

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

    def test_use_to_predict_exception(self):
        with self.assertRaises(NotImplementedError):
            self._cov.use_to_predict = 1.

if __name__ == '__main__':
    unittest.main()
