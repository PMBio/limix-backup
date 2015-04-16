"""LMM testing code"""
import unittest
import scipy as SP
import numpy as np
import sys
from limix.core.covar.freeform import freeform as FreeForm
from limix.core.utils.check_grad import scheck_grad

class TestFreeForm(unittest.TestCase):
    def setUp(self):
        SP.random.seed(1)
        self.n=4
        self.C = FreeForm(self.n)
        self.name = 'freeform'
        self.n_params=self.C.getNumberParams()
        params=SP.randn(self.n_params)
        self.C.setParams(params)

    def test_grad(self):
        params = self.C.getParams()

        for i in xrange(len(params)):

            def set_param(x):
                p = params.copy()
                p[i] = x[0]
                self.C.setParams(p)

            err = scheck_grad(set_param,
                              lambda: np.array([self.C.getParams()[i]]),
                              lambda: self.C.K(),
                              lambda: self.C.Kgrad_param(i))

            np.testing.assert_almost_equal(err, 0.)

if __name__ == '__main__':
    unittest.main()
