"""Variance Decomposition testing code"""
import unittest
import scipy as SP
import scipy.stats
import pdb
import limix

class CVarianceDecomposition_test(unittest.TestCase):
    """test class for CVarianceDecomposition"""

    def genGeno(self):
        self.X  = (SP.rand(self.N,self.S)<0.2)*1.
        self.Kg = SP.dot(self.X,self.X.T)
    
    def genPheno(self):
        dp = SP.ones(self.P); dp[1]=-1
        Y = SP.zeros((self.N,self.P))
        gamma0_g = SP.randn(self.S)
        gamma0_n = SP.randn(self.N)
        for p in range(self.P):
            gamma_g = SP.randn(self.S)
            gamma_n = SP.randn(self.N)
            beta_g  = dp[p]*gamma0_g+gamma_g
            beta_n  = gamma0_n+gamma_n
            y=SP.dot(self.X,beta_g)
            y+=beta_n
            Y[:,p]=y
        self.Y=SP.stats.zscore(Y,0)
    
    def setUp(self):
        SP.random.seed(1)
        self.N = 200
        self.S = 1000
        self.P = 2
        self.genGeno()
        self.genPheno()
        self.vd = limix.CVarianceDecomposition(self.Y)
        self.vd.addFixedEffTerm("specific")
        self.vd.addTerm("Dense",self.Kg)
        self.vd.addTerm("Block",self.Kg)
        self.vd.addTerm("Dense",SP.eye(self.N))
        self.vd.addTerm("Identity",SP.eye(self.N))
        self.vd.initGP()
        
    def test_fit(self):
        """ optimization test """
        self.vd.trainGP()
        params = self.vd.getOptimum()['scales'][:,0]
        params_true = SP.array([-9.44207121e-01,+9.37361082e-01,5.35657525e-01,1.42134130e-08,-1.65923192e-08,3.09197580e-09])
        RV = (params-params_true).max()<1e-6
        self.assertTrue(RV)


if __name__ == '__main__':
    unittest.main()

