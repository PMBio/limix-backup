"""LMM testing code"""
import scipy as SP
import pdb
import sys
import limix
from covar import Acovar_test

class sum_test(Acovar_test):
    """test class for SEcovar"""
    def covar(self):
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

class prod_test(Acovar_test):
    """test class for SEcovar"""
    def covar(self):
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

class kron_test(Acovar_test):
    """test class for SEcovar"""
    def covar(self):
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

if __name__ == '__main__':
    testsum = sum_test()
    testprod= prod_test()
    testkron= kron_test()
    testsum.test_all()
    testprod.test_all()
    testkron.test_all()
