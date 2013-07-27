"""LMM testing code"""
import scipy as SP
import pdb
import sys
import limix
from covar import Acovar_test

class se_test(Acovar_test):
    """test class for SEcovar"""
    def covar(self):
        self.n=10
        self.n_dim=10
        X=SP.rand(self.n,self.n_dim)
        self.C = limix.CCovSqexpARD(self.n_dim)
        self.name = 'CCovSqexpARD'
        self.C.setX(X)

if __name__ == '__main__':
    testse = se_test()
    testse.test_all()
