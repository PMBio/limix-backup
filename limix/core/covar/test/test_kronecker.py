import unittest
import numpy as np
from limix.core.covar import FreeFormCov 
from limix.core.covar import KronCov
from limix.utils.preprocess import covar_rescale
from limix.utils.check_grad import mcheck_grad
import scipy as sp

class TestSumCov(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)
        dim_r = 10
        dim_c = 3
        X = sp.rand(dim_r, dim_r)
        R = covar_rescale(sp.dot(X,X.T))
        C = FreeFormCov(dim_c)
        self._cov = KronCov(C, R)

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

        np.testing.assert_almost_equal(err, 0., decimal=5)

    #def test_Khess(self):
    #
    #    cov = self._cov

    #    for j in range(cov.getNumberParams()):

    #        def func(x, i):
    #            cov.setParams(x)
    #            return cov.K_grad_i(j)

    #        def grad(x, i):
    #            cov.setParams(x)
    #            return cov.K_hess_i_j(j, i)

    #        x0 = cov.getParams()
    #        err = mcheck_grad(func, grad, x0)
    #        np.testing.assert_almost_equal(err, 0.)

if __name__ == '__main__':
    unittest.main()
