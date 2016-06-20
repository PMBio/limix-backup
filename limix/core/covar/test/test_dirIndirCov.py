
import unittest
import scipy as sp
import numpy as np
import sys
from limix.core.covar.dirIndirCov import DirIndirCov
from limix.utils.check_grad import mcheck_grad

class TestDirIndirCov(unittest.TestCase):
    def setUp(self):
        sp.random.seed(1)

        # generate data
        n = 10
        f = 2
        X  = 1.*(sp.rand(n,f)<0.2)
        X -= X.mean(0); X /= X.std(0)
        kinship  = sp.dot(X,X.T)
        kinship /= kinship.diagonal().mean()
        design = sp.zeros((n,n))
        for i in range(n//2):
            design[2*i,2*i+1] = 1
            design[2*i+1,2*i] = 1

        # define covariance
        self.C = DirIndirCov(kinship,design)
        self.C.setRandomParams()

    def test_grad(self):
        def func(x, i):
            self.C.setParams(x)
            return self.C.K()

        def grad(x, i):
            self.C.setParams(x)
            return self.C.K_grad_i(i)

        x0 = self.C.getParams()
        err = mcheck_grad(func, grad, x0)

        np.testing.assert_almost_equal(err, 0., decimal=6)

if __name__ == '__main__':
    unittest.main()
