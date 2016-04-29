"""LMM testing code"""
import unittest
import scipy as SP
import pdb
import sys
import limix.deprecated as dlimix
from .covar import Acovar_test

class CSumCF_test(unittest.TestCase,Acovar_test):
    """test class for CSumCF"""
    def setUp(self):
        SP.random.seed(1)
        self.n=10
        n_dim1=8
        n_dim2=12
        self.C=dlimix.CSumCF()
        self.C.addCovariance(dlimix.CCovSqexpARD(n_dim1));
        self.C.addCovariance(dlimix.CCovLinearARD(n_dim2));
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
        self.C=dlimix.CProductCF()
        self.C.addCovariance(dlimix.CCovSqexpARD(n_dim1));
        self.C.addCovariance(dlimix.CCovLinearARD(n_dim2));
        self.n_dim=self.C.getNumberDimensions()
        X=SP.rand(self.n,self.n_dim)
        self.C.setX(X)
        self.name = 'CProductCF'
        self.n_params=self.C.getNumberParams()
        params=SP.exp(SP.randn(self.n_params))
        self.C.setParams(params)

class CMixedX_test(unittest.TestCase,Acovar_test):
    """test class to ensure that X input free classed (CFixedCF and related) are handled propperly"""

    def setUp(self):
        SP.random.seed(1)
        self.n=10
        n_dim2=12
        K0 = SP.eye(self.n)
        self.C=dlimix.CSumCF()
        #sum of fixed CF and linearARD
        covar1 = dlimix.CFixedCF(K0)
        covar2 = dlimix.CCovLinearARD(n_dim2)
        self.C.addCovariance(covar1)
        self.C.addCovariance(covar2)
        self.n_dim=self.C.getNumberDimensions()
        self.X=SP.rand(self.n,self.n_dim)
        self.C.setX(self.X)
        self.name = 'CSumCF'
        self.n_params=self.C.getNumberParams()
        params=SP.exp(SP.randn(self.n_params))
        self.C.setParams(params)

    def testX_stuff(self):
        """test that X handling is consistent across covariances"""
        self.assertTrue((self.X==self.C.getX()).all())


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
        C1 = dlimix.CCovSqexpARD(n_dim1); C1.setX(X1)
        C2 = dlimix.CCovLinearARD(n_dim2);  C2.setX(X2)
        self.C = dlimix.CKroneckerCF()
        self.C.setRowCovariance(C1)
        self.C.setColCovariance(C2)
        self.n = self.C.Kdim()
        self.n_dim=self.C.getNumberDimensions()
        self.name = 'CKroneckerCF'
        self.n_params=self.C.getNumberParams()
        params=SP.exp(SP.randn(self.n_params))
        self.C.setParams(params)

class CKroneckerCFsoft_test(unittest.TestCase,Acovar_test):
    """test class for CKroneckerCF, when using soft indexes"""
    def setUp(self):
        SP.random.seed(1)
        nr=3
        nc=5
        n_dim1=8
        n_dim2=12
        #truncation of soft kronecker
        self.n_trunk = 10
        Xr = SP.rand(nr,n_dim1)
        Xc = SP.rand(nc,n_dim2)
        Cr = dlimix.CCovSqexpARD(n_dim1); Cr.setX(Xr)
        Cc = dlimix.CCovLinearARD(n_dim2);  Cc.setX(Xc)
        self.C = dlimix.CKroneckerCF()
        self.C.setRowCovariance(Cr)
        self.C.setColCovariance(Cc)
        #set kronecker index
        self.kronecker_index = dlimix.CKroneckerCF.createKroneckerIndex(nc,nr)
        self.n = self.C.Kdim()
        self.n_dim=self.C.getNumberDimensions()
        self.name = 'CKroneckerCF'
        self.n_params=self.C.getNumberParams()
        params=SP.exp(SP.randn(self.n_params))
        self.C.setParams(params)

    def test_kron(self):
        """test that this is a valid Kronecker"""
        self.C.setKroneckerIndicator(SP.zeros([0,0],dtype='int'))
        K1 = self.C.K()
        self.C.setKroneckerIndicator(self.kronecker_index)
        K2 = self.C.K()
        self.assertTrue((K1==K2).all())


    def test_trunk(self):
        """test whether resulting covariance function is truncated"""
        self.C.setKroneckerIndicator(self.kronecker_index[0:self.n_trunk])
        K =self.C.K()
        self.assertTrue(self.C.Kdim()==self.n_trunk)
        self.assertTrue(K.shape[0]==self.n_trunk)


if __name__ == '__main__':
    unittest.main()
