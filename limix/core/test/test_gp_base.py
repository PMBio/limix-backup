import unittest

from limix.core.mean.mean_base import mean_base as lin_mean
from limix.core.covar.sqexp import sqexp
from limix.core.covar.fixed import fixed
from limix.core.covar.combinators import sumcov
from limix.core.gp.gp_base_new import gp as gp_base
from limix.core.utils.check_grad import scheck_grad

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
        errs = []
        for i in xrange(len(gp.getParams()['covar'])):
            def set_param(x):
                params = gp.getParams()
                params['covar'][i] = x[0]
                gp.setParams(params)

            def get_param():
                params = gp.getParams()
                return np.array([params['covar'][i]])

            def get_grad():
                grad = gp.LML_grad()
                return grad['covar'][i]

            err = scheck_grad(set_param,
                              get_param,
                              lambda: gp.LML(),
                              get_grad)

            errs.append(err)

        np.testing.assert_almost_equal(errs, 0., decimal=5)

if __name__ == "__main__":
    unittest.main()
