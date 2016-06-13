"""CLkelihoodISO testing code"""
import unittest
import scipy as SP
import pdb
import limix.deprecated as dlimix
from .covar import Acovar_test

class CLikelihoodISO_test(unittest.TestCase,Acovar_test):
    """test class for CLikelihoodISO"""
    def setUp(self):
        SP.random.seed(1)
        self.n=10
        self.n_dim=10
        X=SP.rand(self.n,self.n_dim)
        self.C = dlimix.CLikNormalIso()
        self.name = 'CLikNormalIso'
        self.C.setX(X)
        self.n_params=self.C.getNumberParams()
        params=SP.exp(SP.randn(self.n_params))
        self.C.setParams(params)

if __name__ == '__main__':
    unittest.main()
