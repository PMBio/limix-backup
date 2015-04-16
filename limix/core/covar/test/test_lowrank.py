"""LMM testing code"""
import unittest
import scipy as sp
import numpy as np
from limix.core.covar.lowrank import lowrank
from limix.core.utils.check_grad import scheck_grad

# class covariance_test(object):
#     """abstract test class for covars"""
#
#     def test_grad(self):
#         """test analytical gradient"""
#         ss = 0
#         for i in range(self.n_params):
#             C_an  = self.C.Kgrad_param(i)
#             C_num = self.C.Kgrad_param_num(i)
#             _ss = ((C_an-C_num)**2).sum()
#             #print i, _ss
#             ss += _ss
#         self.assertTrue(ss<1e-4)

class TestLowRank(unittest.TestCase):
    """test class for CLowRankCF"""
    def setUp(self):
        sp.random.seed(1)
        self.n=4
        self.rank=2
        self.C = lowrank(self.n,self.rank)
        self.name = 'lowrank'
        self.n_params=self.C.getNumberParams()
        params=sp.exp(sp.randn(self.n_params))
        self.C.setParams(params)

    def test_grad(self):
        pass
        # Danilo: I'm assuming that lowrank.py is not finished yet...

        # params = self.C.getParams()
        #
        # for i in xrange(len(params)):
        #     def set_param(x):
        #         p = params.copy()
        #         p[i] = x[0]
        #         self.C.setParams(p)
        #
        #     err = scheck_grad(set_param,
        #                       lambda: np.array([self.C.getParams()[i]]),
        #                       lambda: self.C.K(),
        #                       lambda: self.C.Kgrad_param(i))
        #
        #     np.testing.assert_almost_equal(err, 0.)

if __name__ == '__main__':
    unittest.main()
