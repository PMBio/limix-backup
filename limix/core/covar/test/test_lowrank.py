"""LMM testing code"""
import unittest
import scipy as sp
import numpy as np
from limix.core.covar import LowRankCov
from limix.utils.check_grad import mcheck_grad

class TestLowRank(unittest.TestCase):
    """test class for CLowRankCF"""
    def setUp(self):
        sp.random.seed(1)
        self.n = 4
        self.rank = 2
        self.C = LowRankCov(self.n,self.rank)
        self.name = 'lowrank'
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

        np.testing.assert_almost_equal(err, 0., decimal = 6)

if __name__ == '__main__':
    unittest.main()
