"""LMM testing code"""
import scipy as SP
import pdb
import sys
import limix
from covar import Acovar_test

class linISO_test(Acovar_test):
    """test class for SEcovar"""
    def covar(self):
        self.n=10
        self.n_dim=10
        X=SP.rand(self.n,self.n_dim)
        self.C = limix.CCovLinearISO(self.n_dim)
        self.name = 'CCovLinearISO'
        self.C.setX(X)

class linARD_test(Acovar_test):
    """test class for SEcovar"""
    def covar(self):
        self.n=10
        self.n_dim=10
        X=SP.rand(self.n,self.n_dim)
        self.C = limix.CCovLinearARD(self.n_dim)
        self.name = 'CCovLinearARD'
        self.C.setX(X)

if __name__ == '__main__':
    testlinISO = linISO_test()
    testlinARD = linARD_test()
    testlinISO.test_all()
    testlinARD.test_all()
