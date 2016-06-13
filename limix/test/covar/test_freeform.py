"""LMM testing code"""
import unittest
import scipy as SP
import pdb
import limix.deprecated as dlimix
from .covar import Acovar_test

class CFreeFormCF_test(unittest.TestCase,Acovar_test):
    """test class for CFreeFormCF"""
    def setUp(self):
        SP.random.seed(1)
        self.n=4
        self.C = dlimix.CFreeFormCF(self.n)
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
        self.C = dlimix.CRankOneCF(self.n)
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
        self.C = dlimix.CLowRankCF(self.n,self.rank)
        self.name = 'CLowRankCF'
        self.n_params=self.C.getNumberParams()
        params=SP.exp(SP.randn(self.n_params))
        self.C.setParams(params)

class CDiagonalCF_test(unittest.TestCase,Acovar_test):
    """test class for CDiagonalCF"""
    def setUp(self):
        SP.random.seed(1)
        self.n=2
        self.C = dlimix.CDiagonalCF(self.n)
        self.name = 'CDiagonalCF'
        self.n_params=self.C.getNumberParams()
        params=SP.exp(SP.randn(self.n_params))
        self.C.setParams(params)

class CFixedCF_test(unittest.TestCase,Acovar_test):
    """test class for CFixedCF"""
    def setUp(self):
        SP.random.seed(1)
        self.n=4
        self.C = dlimix.CFixedCF(SP.ones((self.n,self.n)))
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
        self.C = dlimix.CRank1diagCF(self.n)
        self.name = 'CRank1diagCF'
        self.n_params=self.C.getNumberParams()
        params=SP.exp(SP.randn(self.n_params))
        self.C.setParams(params)

class CPolyCF_test(unittest.TestCase,Acovar_test):
    """test class for CRank1diagCF"""
    def setUp(self):
        SP.random.seed(1)
        self.n=4
        self.d=2
        self.K=3
        self.C = dlimix.CPolyCF(self.n,self.d,self.K)
        self.name = 'CPolyCF'
        self.n_params=self.C.getNumberParams()
        params=SP.randn(self.n_params)
        self.C.setParams(params)

class CFixedDiagonalCF_test(unittest.TestCase,Acovar_test):
    """test class for CFixedDiagonalCF"""
    def setUp(self):
        SP.random.seed(1)
        self.n=4
        self.rank=1
        d = SP.rand(self.n)+1
        C0 = dlimix.CSumCF()
        C0.addCovariance(dlimix.CLowRankCF(self.n,self.rank))
        C0.addCovariance(dlimix.CDiagonalCF(self.n))
        self.C = dlimix.CFixedDiagonalCF(C0,d)
        self.name = 'CFixedDiagonalCF'
        self.n_params=self.C.getNumberParams()
        params=SP.randn(self.n_params)
        self.C.setParams(params)
        """
        h = 1e-4
        e = SP.zeros(self.n_params)
        for i in range(self.n_params):
            print i
            e[i] = 1
            self.C.setParams(params)
            Cgrad = self.C.Kgrad_param(i)
            self.C.setParams(params+h*e)
            Cr = self.C.K()
            self.C.setParams(params-h*e)
            Cl = self.C.K()
            Cgrad1 = (Cr-Cl)/(2*h)
            print Cgrad/Cgrad1
            pdb.set_trace()
            e[i] = 0
        """

if __name__ == '__main__':
    unittest.main()
