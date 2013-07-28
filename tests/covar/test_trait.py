"""LMM testing code"""
import scipy as SP
import pdb
import sys
import limix
from covar import Acovar_test

class FreeForm_test(Acovar_test):
    """test class for SEcovar"""
    def covar(self):
        self.n=4
        self.C = limix.CTFreeForm(self.n)
        self.name = 'CTFreeForm'

class Dense_test(Acovar_test):
    """test class for SEcovar"""
    def covar(self):
        self.n=4
        self.C = limix.CTDense(self.n)
        self.name = 'CTDense'

class Diagonal_test(Acovar_test):
    """test class for SEcovar"""
    def covar(self):
        self.n=4
        self.C = limix.CTDiagonal(self.n)
        self.name = 'CTDiagonal'

class Fixed_test(Acovar_test):
    """test class for SEcovar"""
    def covar(self):
        self.n=4
        self.C = limix.CTFixed(self.n,SP.ones((self.n,self.n)))
        self.name = 'CTFixed'

class LowRank_test(Acovar_test):
    """test class for SEcovar"""
    def covar(self):
        self.n=4
        self.C = limix.CTLowRank(self.n)
        self.name = 'CTLowRank'

if __name__ == '__main__':
    testFreeForm = FreeForm_test()
    #testDense = Dense_test()
    #testDiagonal = Diagonal_test()
    #testFixed = Fixed_test()
    #testLowRank = LowRank_test()
    testFreeForm.test_all()
    #testDense.test_all()
    #testDiagonal.test_all()
    #testFixed.test_all()
    #testLowRank.test_all()
