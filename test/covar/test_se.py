"""LMM testing code"""
import unittest
import scipy as SP
import pdb
import limix
from covar import Acovar_test

class CCovSqexpARD_test(unittest.TestCase,Acovar_test):
    """test class for CCovSqexpARD"""
    def setUp(self):
        SP.random.seed(1)
        self.n=10
        self.n_dim=10
        X=SP.rand(self.n,self.n_dim)
        self.C = limix.CCovSqexpARD(self.n_dim)
        self.name = 'CCovSqexpARD'
        self.C.setX(X)
        K = self.C.K()
        self.n_params=self.C.getNumberParams()
        params=SP.exp(SP.randn(self.n_params))
        self.C.setParams(params)

if __name__ == '__main__':
    unittest.main()
