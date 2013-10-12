"""LMM testing code"""
import unittest
import scipy as SP
import pdb
import limix
from covar import Acovar_test

class CFreeFormCF_test(unittest.TestCase,Acovar_test):
    """test class for CFreeFormCF"""
    def setUp(self):
        SP.random.seed(1)
        self.n=4
        self.C = limix.CFreeFormCF(self.n)
        self.name = 'CFreeFormCF'
        self.n_params=self.C.getNumberParams()
        K = self.C.K()
        params=SP.exp(SP.randn(self.n_params))
        self.C.setParams(params)

class CRankOneCF_test(unittest.TestCase,Acovar_test):
    """test class for CRankOneCF"""
    def setUp(self):
        SP.random.seed(1)
        self.n=4
        self.C = limix.CRankOneCF(self.n)
        self.name = 'CRankOneCF'
        self.n_params=self.C.getNumberParams()
        K = self.C.K()
        params=SP.exp(SP.randn(self.n_params))
        self.C.setParams(params)

class CLowRankCF_test(unittest.TestCase,Acovar_test):
    """test class for CLowRankCF"""
    def setUp(self):
        SP.random.seed(1)
        self.n=4
        self.rank=2
        self.C = limix.CLowRankCF(self.n,self.rank)
        self.name = 'CLowRankCF'
        self.n_params=self.C.getNumberParams()
        params=SP.exp(SP.randn(self.n_params))
        self.C.setParams(params)

class CDiagonalCF_test(unittest.TestCase,Acovar_test):
    """test class for CDiagonalCF"""
    def setUp(self):
        SP.random.seed(1)
        self.n=2
        self.C = limix.CDiagonalCF(self.n)
        self.name = 'CDiagonalCF'
        self.n_params=self.C.getNumberParams()
        params=SP.exp(SP.randn(self.n_params))
        self.C.setParams(params)

class CFixedCF_test(unittest.TestCase,Acovar_test):
    """test class for CFixedCF"""
    def setUp(self):
        SP.random.seed(1)
        self.n=4
        self.C = limix.CFixedCF(SP.ones((self.n,self.n)))
        self.name = 'CFixedCF'
        self.n_params=self.C.getNumberParams()
        K = self.C.K()        
        params=SP.exp(SP.randn(self.n_params))
        self.C.setParams(params)

class CRank1diagCF_test(unittest.TestCase,Acovar_test):
    """test class for CRank1diagCF"""
    def setUp(self):
        SP.random.seed(1)
        self.n=4
        self.C = limix.CRank1diagCF(self.n)
        self.name = 'CRank1diagCF'
        self.n_params=self.C.getNumberParams()
        params=SP.exp(SP.randn(self.n_params))
        self.C.setParams(params)

if __name__ == '__main__':
    unittest.main()
