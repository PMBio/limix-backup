"""CLkelihoodISO testing code"""
import scipy as SP
import pdb
import sys
import limix
from covar import Acovar_test

class likISO_test(Acovar_test):
    """test class for CLikelihoodISO"""
    def covar(self):
        self.n=10
        self.n_dim=10
        X=SP.rand(self.n,self.n_dim)
        self.C = limix.CLikNormalIso()
        self.name = 'CLikNormalIso'
        self.C.setX(X)

if __name__ == '__main__':
    testlikISO = likISO_test()
    testlikISO.test_all()
