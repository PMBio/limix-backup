"""LMM testing code"""
import scipy as SP
import pdb
import sys
import limix
from covar import Acovar_test

class fixed_test(Acovar_test):
    """test class for SEcovar"""
    def covar(self):
        self.n=10
        K=SP.rand(self.n,self.n)
        self.C = limix.CFixedCF(K)
        self.name = 'CFixedCF'

if __name__ == '__main__':
    testfixed = fixed_test()
    testfixed.test_all()
