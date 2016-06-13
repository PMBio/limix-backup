import unittest
from limix.core.covar import FreeFormCov
from limix.core.mean import MeanKronSum
from limix.core.gp import GP2KronSum
from limix.core.gp import GP
from limix.utils.preprocess import covar_rescale
from limix.utils.check_grad import mcheck_grad

import numpy as np
import scipy as sp
import pdb


class TestGPBase(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)

        # define phenotype
        N = 200
        P = 2
        self.Y = sp.randn(N, P)
        # define fixed effects
        self.F = []; self.A = []
        self.F.append(1.*(sp.rand(N,2)<0.5))
        self.A.append(sp.eye(P))
        # define row covariance
        f = 10
        X = 1.*(sp.rand(N, f)<0.2)
        self.R  = covar_rescale(sp.dot(X,X.T))
        self.R += 1e-4 * sp.eye(N)
        # define col covariances
        self.Cg = FreeFormCov(P)
        self.Cn = FreeFormCov(P)
        self.Cg.setCovariance(0.5 * sp.cov(self.Y.T))
        self.Cn.setCovariance(0.5 * sp.cov(self.Y.T))
        # define gp
        self.gp = GP2KronSum(Y=self.Y, F=self.F, A=self.A, Cg=self.Cg,
                             Cn=self.Cn, R=self.R)

    @unittest.skip("someone has to fix it")
    def test_grad(self):

        gp = self.gp

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
        np.testing.assert_almost_equal(err, 0., decimal=4)

    def test_grad_activation(self):

        gp = self.gp

        self.Cg._K_act = False

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
        np.testing.assert_almost_equal(err, 0., decimal=4)

        self.Cg._K_act = True
        self.Cn._K_act = False

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
        np.testing.assert_almost_equal(err, 0., decimal=4)

    def test_correct_inputs(self):
        np.asarray(None, dtype=float)
        # self.gp = GP2KronSum(Y=self.Y, F=self.F, A=self.A, Cg=self.Cg,
        #                      Cn=self.Cn, R=self.R)

if __name__ == "__main__":
    unittest.main()
