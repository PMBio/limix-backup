import unittest
import numpy as np
from limix.core.mean.mean_base import MeanBase as lin_mean
from limix.core.covar.sqexp import sqexp
from limix.core.covar.fixed import fixed
from limix.core.covar.combinators import sumcov
from limix.core.gp.gp_base_new import gp as gp_base

class TestGP(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)
        N = 400
        s_x = 0.05
        s_y = 0.1
        X = (np.linspace(0,2,N)+s_x*np.random.randn(N))[:,np.newaxis]
        Y = np.sin(X)+s_y*np.random.randn(N,1)
        #pl.plot(x,y,'x')

        # define mean term
        F = 1.*(np.random.rand(N,2)<0.2)
        mean = lin_mean(Y,F)

        # define covariance matrices
        covar1 = sqexp(X)
        covar2 = fixed(np.eye(N))
        covar  = sumcov(covar1,covar2)

        # self._gp = gp_base(covar=covar,mean=mean)
        # covar.setRandomParams()

    def test_lml(self):
        return
        # self._gp.LML()

    def test_lml_grad(self):
        return
        # self._gp.LML_grad()

if __name__ == '__main__':
    unittest.main()
