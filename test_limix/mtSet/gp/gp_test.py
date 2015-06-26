import sys
sys.path.append('./..')
sys.path.insert(0,'./../..')
import unittest
from unitest_class import unittestClass
import os
import scipy as SP
import pdb
from limix.core.mean import mean
from limix.core.covar import freeform
from limix.core.gp.gp2kronSum import gp2kronSum
import limix.core.optimize.optimize_bfgs as OPT

#path_abs = os.path.dirname(os.path.abspath(sys.argv[0]))
#path_mtSet = os.path.join(path_abs,'../..')

class gp_unittest(unittestClass):
    """test class for optimization""" 
    
    def setUp(self):
        SP.random.seed(0)
        self.Y = SP.loadtxt('./data/Y.txt') 
        self.XX = SP.loadtxt('./data/XX.txt') 
        self.Xr = SP.loadtxt('./data/Xr.txt') 
        self.N,self.P = self.Y.shape
        self.write = False 

    def test_gp2kronSumOpt(self):
        fbasename = 'gp2kronSumOpt'
        mu = mean(self.Y)
        mu.addFixedEffect(F=self.Xr[:,0:2],A=SP.ones((1,self.P)))
        mu.addFixedEffect(F=self.Xr[:,2:4],A=SP.eye(self.P))
        Cg = freeform(self.P)
        Cn = freeform(self.P)
        gp = gp2kronSum(mu,Cg,Cn,XX=self.XX)
        params0 = {'Cg': SP.randn(Cg.getParams().shape[0]),
                   'Cn': SP.randn(Cg.getParams().shape[0])}
        conv,info = OPT.opt_hyper(gp,params0,factr=1e3)
        ext = {'Cg':gp.Cg.K(),'Cn':gp.Cn.K()}
        if self.write: self.saveStuff(fbasename,ext)
        RV = self.assess(fbasename,ext)
        self.assertTrue(RV)

if __name__ == '__main__':
    unittest.main()

