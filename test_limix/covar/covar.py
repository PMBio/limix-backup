"""LMM testing code"""
import unittest
import scipy as SP
import pdb
import sys
import limix


class Acovar_test(object):
    """abstract test class for covars"""

    def test_grad(self):
        """test analytical gradient"""
        RV = self.C.check_covariance_Kgrad_theta(self.C)
        self.assertTrue(RV)

    # def test_hess(self):
    #     """test analytical hessian"""
    #     D2=SP.zeros((self.n_params,self.n_params))
    #     for i in range(self.n_params):
    #         for j in range(self.n_params):
    #             D2[i,j]=((self.C.Khess_param(i,j)-self.C.Khess_param_num(self.C,i,j))**2).max()
    #     RV=D2.max()<1E-6
    #     self.assertTrue(RV)
