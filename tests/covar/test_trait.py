"""LMM testing code"""
import unittest
import scipy as SP
import pdb
import limix
from covar import Acovar_test

class CTFreeForm_test(unittest.TestCase,Acovar_test):
    """test class for CTFreeForm"""
    def setUp(self):
        SP.random.seed(1)
        self.n=4
        self.C = limix.CFreeFormCF(self.n)
        self.name = 'CFreeFormCF'
        self.n_params=self.C.getNumberParams()
        params=SP.exp(SP.randn(self.n_params))
        self.C.setParams(params)

"""
class Dense_test(unittest.TestCase,Acovar_test):
    def covar(self):
        self.n=4
        self.C = limix.CTDense(self.n)
        self.name = 'CTDense'

class Diagonal_test(unittest.TestCase,Acovar_test):
    def covar(self):
        self.n=4
        self.C = limix.CTDiagonal(self.n)
        self.name = 'CTDiagonal'

class Fixed_test(unittest.TestCase,Acovar_test):
    def covar(self):
        self.n=4
        self.C = limix.CTFixed(self.n,SP.ones((self.n,self.n)))
        self.name = 'CTFixed'

class LowRank_test(unittest.TestCase,Acovar_test):
    def covar(self):
        self.n=4
        self.C = limix.CTLowRank(self.n)
        self.name = 'CTLowRank'
"""

if __name__ == '__main__':
    unittest.main()
