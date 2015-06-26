import unittest
from limix.core.covar import FreeFormCov
from limix.core.mean import MeanKronSum
from limix.core.gp import GP2KronSumLR
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
        # define fixed effects
        F = []; A = []
        F.append(1.*(sp.rand(N,2)<0.5))
        A.append(sp.eye(P))
        # define row caoriance
        f = 10
        G = 1.*(sp.rand(N, f)<0.2)
        # define col covariances
        Cr = FreeFormCov(P)
        self._Cr = Cr
        Cn = FreeFormCov(P)
        Cr.setCovariance(0.5 * sp.cov(Y.T))
        Cn.setCovariance(0.5 * sp.cov(Y.T))
        # define gp
        self.gp = GP2KronSumLR(Y = Y, F = F, A = A, Cn = Cn, G = G)

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

        # gp.covar.act_Cr = False
        self._Cr.act_K = False

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
