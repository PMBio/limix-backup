"""LMM testing code"""
import unittest
import scipy as SP
import pdb
import limix
from covar import Acovar_test

class CFixedCF_test(unittest.TestCase,Acovar_test):
    """test class for CFixedCF"""
    def setUp(self):
        SP.random.seed(1)
        self.n=10
        K=SP.rand(self.n,self.n)
        self.C = limix.CFixedCF(K)
        self.name = 'CFixedCF'
        self.n_params=self.C.getNumberParams()
        params=SP.exp(SP.randn(self.n_params))
        self.C.setParams(params)

if __name__ == '__main__':
    unittest.main()
