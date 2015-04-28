import unittest

from limix.core.mean.mean_base import mean_base as lin_mean
from limix.core.covar.sqexp import sqexp
from limix.core.covar.fixed import fixed
from limix.core.covar.combinators import sumcov
from limix.core.gp.gp_base import gp as gp_base
from limix.core.utils.check_grad import mcheck_grad

import numpy as np
import scipy as sp


class TestGPBase(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)

        # generate data
        N = 400
        s_x = 0.05
        s_y = 0.1
        X = (sp.linspace(0,2,N)+s_x*sp.randn(N))[:,sp.newaxis]
        Y = sp.sin(X)+s_y*sp.randn(N,1)
        Y-= Y.mean(0)
        Y/= Y.std(0)

        Xstar = sp.linspace(0,2,1000)[:,sp.newaxis]

        # define mean term
        F = 1.*(sp.rand(N,2)<0.2)
        mean = lin_mean(Y,F)

        # define covariance matrices
        covar1 = sqexp(X,Xstar=Xstar)
        covar2 = fixed(sp.eye(N))
        covar  = sumcov(covar1,covar2)

        # define gp
        self._gp = gp_base(covar=covar,mean=mean)

    def test_grad(self):

        gp = self._gp

        def func(x, i):
            params = gp.getParams()
            params['covar'] = x
            gp.setParams(params)
            return gp.LML()

        def grad(x, i):
            params = gp.getParams()
            params['covar'] = x
            gp.setParams(params)
            grad = gp.LML_grad()
            return grad['covar'][i]

        x0 = gp.getParams()['covar']
        err = mcheck_grad(func, grad, x0)
        np.testing.assert_almost_equal(err, 0., decimal=6)

if __name__ == "__main__":
    unittest.main()
