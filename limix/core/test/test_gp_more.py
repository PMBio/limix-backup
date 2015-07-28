import unittest

from limix.core.mean import MeanKronSum
from limix.core.covar import FreeFormCov
from limix.core.covar import FixedCov
from limix.core.covar import KronCov 
from limix.core.covar.combinators import SumCov
from limix.core.gp.gp_base import GP
from limix.core.gp import GP2KronSum
from limix.utils.check_grad import mcheck_grad
from limix.utils.preprocess import covar_rescale

import copy
import numpy as np
import scipy as sp


class TestGPBase(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)

        # define phenotype
        N = 10 
        P = 3
        Y = sp.randn(N,P)

        # pheno with missing data
        Ym = Y.copy()
        Im = sp.rand(N, P)<0.2
        Ym[Im] = sp.nan

        # define fixed effects
        F = []; A = []
        F.append(1.*(sp.rand(N,2)<0.5))
        A.append(sp.eye(P))
        mean = MeanKronSum(Y, F=F, A=A)
        mean_m = MeanKronSum(Ym, F=F, A=A)

        # define row caoriance
        f = 10
        X = 1.*(sp.rand(N, f)<0.2)
        R = covar_rescale(sp.dot(X,X.T))
        R+= 1e-4 * sp.eye(N)

        # define col covariances
        Cg = FreeFormCov(P)
        Cn = FreeFormCov(P)
        Cg.setRandomParams()
        Cn.setRandomParams()

        # define covariance matrices
        covar1 = KronCov(Cg, R)
        covar2 = KronCov(Cn, sp.eye(N))
        covar  = SumCov(covar1,covar2)

        # define covariance matrice with missing data
        Iok = (~Im).reshape(N * P, order='F')
        covar1_m = KronCov(copy.copy(Cg), R, Iok=Iok)
        covar2_m = KronCov(copy.copy(Cn), sp.eye(N), Iok=Iok)
        covar_m  = SumCov(covar1_m,covar2_m)

        # define gp
        self._gp = GP(covar=covar, mean=mean)
        self._gpm = GP(covar=covar_m, mean=mean_m)
        self._gp2ks = GP2KronSum(Y=Y, F=F, A=A, Cg=Cg, Cn=Cn, R=R) 

    def test_gpbase_gp2kronSum(self): 
        d1 = self._gp2ks.LML()-self._gp.LML()
        d2 = self._gp2ks.LML_grad()['covar']
        d2-= self._gp.LML_grad()['covar']
        d  = sp.concatenate([sp.array([d1]), d2])
        np.testing.assert_almost_equal(d, 0., decimal=8)
        

    def test_grad_gpbase_missdata(self):

        gp = self._gpm

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
        np.testing.assert_almost_equal(err, 0., decimal=5)

if __name__ == "__main__":
    unittest.main()
