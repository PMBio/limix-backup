import unittest
import numpy as np
from limix.core.covar.sqexp import sqexp as SQExp
from limix.core.utils.check_grad import scheck_grad
import scipy as sp

class TestSQExp(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)
        self._X = np.random.randn(10, 5)
        self._cov = SQExp(self._X)

    def test_setX_retE(self):
        X1 = self._X
        np.random.seed(2)
        X2 = np.random.randn(10, 5)

        E1 = sp.spatial.distance.pdist(X1,'euclidean')**2
        E1 = sp.spatial.distance.squareform(E1)

        E2 = sp.spatial.distance.pdist(X2,'euclidean')**2
        E2 = sp.spatial.distance.squareform(E2)

        np.testing.assert_almost_equal(E1, self._cov.E())

        self._cov.X = X2
        np.testing.assert_almost_equal(E2, self._cov.E())

    def test_Kgrad(self):
        def set_scale(x):
            self._cov.scale = x[0]
        err = scheck_grad(set_scale,
                          lambda: np.array([self._cov.scale]),
                          lambda: self._cov.K(),
                          lambda: self._cov.K_grad_i(0))

        np.testing.assert_almost_equal(err, 0.)

        def set_length(x):
            self._cov.length = x[0]
        err = scheck_grad(set_length,
                          lambda: np.array([self._cov.length]),
                          lambda: self._cov.K(),
                          lambda: self._cov.K_grad_i(1))

        np.testing.assert_almost_equal(err, 0.)


if __name__ == '__main__':
    unittest.main()
