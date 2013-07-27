"""LMM testing code"""
import scipy as SP
import pdb
import sys
import limix

sys.path.append('./../helper/')
from helper import message


class Acovar_test:
    """test class for SEcovar"""
    
    def __init__(self):
        self.generate()

    def generate(self):
        SP.random.seed(1)
        self.covar()
        self.n_params=self.C.getNumberParams()
        params=SP.exp(SP.randn(self.n_params))
        self.C.setParams(params)

    def test_grad(self):
        """test analytical gradient"""
        RV = self.C.check_covariance_Kgrad_theta(self.C)
        print '   ...gradient %s' % message(RV)

    def test_hess(self):
        """test analytical hessian"""
        D2=SP.zeros((self.n_params,self.n_params))
        for i in range(self.n_params):
            for j in range(self.n_params):
                D2[i,j]=((self.C.Khess_param(i,j)-self.C.Khess_param_num(self.C,i,j))**2).max()
        RV=D2.max()<1E-6
        print '   ...hessian %s' % message(RV)
        
    def test_all(self):
        print '...testing %s' % self.name
        self.test_grad()
        self.test_hess()

