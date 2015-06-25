import unittest
from limix.core.covar import FreeFormCov
from limix.core.mean import MeanKronSum
from limix.core.gp import GP3KronSumLR
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
        Y = sp.randn(N,P)
        # define row caoriance
        f = 10
        G = 1.*(sp.rand(N, f)<0.2)
        X = 1.*(sp.rand(N, f)<0.2)
        R = covar_rescale(sp.dot(X,X.T))
        R+= 1e-4 * sp.eye(N)
        # define col covariances
        Cg = FreeFormCov(P)
        self._Cg = Cg
        Cn = FreeFormCov(P)
        Cg.setCovariance(0.5 * sp.cov(Y.T))
        Cn.setCovariance(0.5 * sp.cov(Y.T))
        # define gp
        self.gp = GP3KronSumLR(Y = Y, Cg = Cg, Cn = Cn, R = R, G = G, rank = 1)

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

        self._Cg._K_act = False

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

if __name__ == "__main__":
    unittest.main()
