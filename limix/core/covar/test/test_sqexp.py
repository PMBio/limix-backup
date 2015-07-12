import unittest
import numpy as np
from limix.core.covar import SQExpCov
from limix.utils.check_grad import mcheck_grad
from limix.core.type.exception import NotArrayConvertibleError
import scipy as sp

class TestSQExp(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)
        self._X = np.random.randn(10, 5)
        self._cov = SQExpCov(self._X)

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

    def test_param_activation(self):
        self._cov.act_scale = False
        self._cov.act_length = False
        self.assertEqual(len(self._cov.getParams()), 0)

        self._cov.act_scale = False
        self._cov.act_length = True
        self.assertEqual(len(self._cov.getParams()), 1)

        self._cov.act_scale = True
        self._cov.act_length = False
        self.assertEqual(len(self._cov.getParams()), 1)

        self._cov.act_scale = True
        self._cov.act_length = True
        self.assertEqual(len(self._cov.getParams()), 2)

        self._cov.act_scale = False
        self._cov.act_length = False
        self._cov.setParams(np.array([]))
        with self.assertRaises(ValueError):
            self._cov.setParams(np.array([0]))

        with self.assertRaises(ValueError):
            self._cov.K_grad_i(0)

        with self.assertRaises(ValueError):
            self._cov.K_grad_i(1)

    def test_Kgrad(self):

        def func(x, i):
            self._cov.scale = x[i]
            return self._cov.K()

        def grad(x, i):
            self._cov.scale = x[i]
            return self._cov.K_grad_i(0)

        x0 = np.array([self._cov.scale])
        err = mcheck_grad(func, grad, x0)

        np.testing.assert_almost_equal(err, 0.)

        def func(x, i):
            self._cov.length = x[i]
            return self._cov.K()

        def grad(x, i):
            self._cov.scale = x[i]
            return self._cov.K_grad_i(1)

        x0 = np.array([self._cov.scale])
        err = mcheck_grad(func, grad, x0)

    def test_Kgrad_activation(self):
        self._cov.act_length = False
        def func(x, i):
            self._cov.scale = x[i]
            return self._cov.K()

        def grad(x, i):
            self._cov.scale = x[i]
            return self._cov.K_grad_i(0)

        x0 = np.array([self._cov.scale])
        err = mcheck_grad(func, grad, x0)

        np.testing.assert_almost_equal(err, 0.)

        self._cov.act_scale = False
        self._cov.act_length = True
        def func(x, i):
            self._cov.length = x[i]
            return self._cov.K()

        def grad(x, i):
            self._cov.length = x[i]
            return self._cov.K_grad_i(0)

        x0 = np.array([self._cov.length])
        err = mcheck_grad(func, grad, x0)

        np.testing.assert_almost_equal(err, 0.)

    def test_Khess(self):

        def func(x, i):
            self._cov.scale = x[i]
            return self._cov.K_grad_i(0)

        def grad(x, i):
            self._cov.scale = x[i]
            return self._cov.K_hess_i_j(0, 0)

        x0 = np.array([self._cov.scale])
        err = mcheck_grad(func, grad, x0)
        np.testing.assert_almost_equal(err, 0., decimal=5)

        def func(x, i):
            self._cov.length = x[i]
            return self._cov.K_grad_i(0)

        def grad(x, i):
            self._cov.length = x[i]
            return self._cov.K_hess_i_j(0, 1)

        x0 = np.array([self._cov.scale])
        err = mcheck_grad(func, grad, x0)
        np.testing.assert_almost_equal(err, 0., decimal=5)

        def func(x, i):
            self._cov.length = x[i]
            return self._cov.K_grad_i(1)

        def grad(x, i):
            self._cov.length = x[i]
            return self._cov.K_hess_i_j(1, 1)

        x0 = np.array([self._cov.scale])
        err = mcheck_grad(func, grad, x0)
        np.testing.assert_almost_equal(err, 0., decimal=5)


    def test_input(self):
        with self.assertRaises(ValueError):
            SQExpCov(np.array([[np.inf]]))

        with self.assertRaises(ValueError):
            SQExpCov(np.array([[np.nan]]))

        with self.assertRaises(NotArrayConvertibleError):
            SQExpCov("Ola meu querido.")

if __name__ == '__main__':
    unittest.main()
