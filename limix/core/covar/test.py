"""LMM testing code"""
import unittest
import scipy as SP
import sys
from freeform import freeform
from lowrank import lowrank

class covariance_test(object):
    """abstract test class for covars"""

    def test_grad(self):
        """test analytical gradient"""
        ss = 0
        for i in range(self.n_params):
            C_an  = self.C.Kgrad_param(i)
            C_num = self.C.Kgrad_param_num(i)
            _ss = ((C_an-C_num)**2).sum()
            #print i, _ss
            ss += _ss 
        self.assertTrue(ss<1e-4)

class CFreeFormCF_test(unittest.TestCase,covariance_test):
    """test class for CFreeFormCF"""
    def setUp(self):
        SP.random.seed(1)
        self.n=4
        self.C = freeform(self.n)
        self.name = 'freeform'
        self.n_params=self.C.getNumberParams()
        params=SP.randn(self.n_params)
        self.C.setParams(params)

class CLowRankCF_test(unittest.TestCase,covariance_test):
    """test class for CLowRankCF"""
    def setUp(self):
        SP.random.seed(1)
        self.n=4
        self.rank=2
        self.C = lowrank(self.n,self.rank)
        self.name = 'lowrank'
        self.n_params=self.C.getNumberParams()
        params=SP.exp(SP.randn(self.n_params))
        self.C.setParams(params)

if __name__ == '__main__':
    unittest.main()

