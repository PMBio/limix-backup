import unittest
import numpy as np
from limix.core.covar.sqexp import sqexp as SQExp
from limix.core.covar.combinators import sumcov as SumCov
from limix.core.utils.check_grad import scheck_grad
import scipy as sp

class TestSumCov(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)
        self._X1 = np.random.randn(10, 5)
        self._X2 = np.random.randn(10, 8)
        self._cov1 = SQExp(self._X1)
        self._cov2 = SQExp(self._X2)
        self._cov = SumCov(self._cov1, self._cov2)

    def test_sum_combination(self):
        K1 = self._cov1.K() + self._cov2.K()
        K2 = self._cov.K()

        np.testing.assert_almost_equal(K1, K2)

    def test_Kgrad(self):

        cov = self._cov

        errs = []
        for i in xrange(len(cov.getParams())):
            def set_param(x):
                params = cov.getParams()
                params[i] = x[0]
                cov.setParams(params)

            err = scheck_grad(set_param,
                              lambda: np.asarray([cov.getParams()[i]]),
                              lambda: cov.K(),
                              lambda: cov.K_grad_i(i))
            errs.append(err)

        np.testing.assert_almost_equal(errs, 0.)

if __name__ == '__main__':
    unittest.main()
