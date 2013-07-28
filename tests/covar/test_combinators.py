"""LMM testing code"""
import unittest
import scipy as SP
import pdb
import sys
sys.path.append('./../release.darwin/interfaces/python/')
import limix
from covar import Acovar_test

class CSumCF_test(unittest.TestCase,Acovar_test):
    """test class for CSumCF"""
    def setUp(self):
        SP.random.seed(1)
        self.n=10
        n_dim1=8
        n_dim2=12
        self.C=limix.CSumCF()
        self.C.addCovariance(limix.CCovSqexpARD(n_dim1));
        self.C.addCovariance(limix.CCovLinearARD(n_dim2));
        self.n_dim=self.C.getNumberDimensions()
        X=SP.rand(self.n,self.n_dim)
        self.C.setX(X)
        self.name = 'CSumCF'
        self.n_params=self.C.getNumberParams()
        params=SP.exp(SP.randn(self.n_params))
        self.C.setParams(params)

class CProductCF_test(unittest.TestCase,Acovar_test):
    """test class for CProductCF"""
    def setUp(self):
        SP.random.seed(1)
        self.n=10
        n_dim1=8
        n_dim2=12
        self.C=limix.CProductCF()
        self.C.addCovariance(limix.CCovSqexpARD(n_dim1));
        self.C.addCovariance(limix.CCovLinearARD(n_dim2));
        self.n_dim=self.C.getNumberDimensions()
        X=SP.rand(self.n,self.n_dim)
        self.C.setX(X)
        self.name = 'CProductCF'
        self.n_params=self.C.getNumberParams()
        params=SP.exp(SP.randn(self.n_params))
        self.C.setParams(params)

class CKroneckerCF_test(unittest.TestCase,Acovar_test):
    """test class for CKroneckerCF"""
    def setUp(self):
        SP.random.seed(1)
        n1=3
        n2=5
        n_dim1=8
        n_dim2=12
        X1 = SP.rand(n1,n_dim1)
        X2 = SP.rand(n2,n_dim2)
        C1 = limix.CCovSqexpARD(n_dim1); C1.setX(X1)
        C2 = limix.CCovLinearARD(n_dim2);  C2.setX(X2)
        self.C = limix.CKroneckerCF()
        self.C.setRowCovariance(C1)
        self.C.setColCovariance(C2)
        self.n = self.C.Kdim()
        self.n_dim=self.C.getNumberDimensions()
        self.name = 'CKroneckerCF'
        self.n_params=self.C.getNumberParams()
        params=SP.exp(SP.randn(self.n_params))
        self.C.setParams(params)

if __name__ == '__main__':
    unittest.main()
