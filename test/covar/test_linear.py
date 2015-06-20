"""LMM testing code"""
import unittest
import scipy as SP
import pdb
import limix
from covar import Acovar_test

class CCovLinearISO_test(unittest.TestCase,Acovar_test):
    """test class for CCovLinearISO"""
    def setUp(self):
        SP.random.seed(1)
        self.n=10
        self.n_dim=10
        X=SP.rand(self.n,self.n_dim)
        self.C = limix.CCovLinearISO(self.n_dim)
        self.name = 'CCovLinearISO'
        self.C.setX(X)
        self.n_params=self.C.getNumberParams()
        K = self.C.K()
        params=SP.exp(SP.randn(self.n_params))
        self.C.setParams(params)

class CCovLinearARD_test(unittest.TestCase,Acovar_test):
    """test class for CCovLinearARD"""
    def setUp(self):
        SP.random.seed(1)
        self.n=10
        self.n_dim=10
        X=SP.rand(self.n,self.n_dim)
        self.C = limix.CCovLinearARD(self.n_dim)
        self.name = 'CCovLinearARD'
        self.C.setX(X)
        self.n_params=self.C.getNumberParams()
        K = self.C.K()
        params=SP.exp(SP.randn(self.n_params))
        self.C.setParams(params)

if __name__ == '__main__':
    unittest.main()
