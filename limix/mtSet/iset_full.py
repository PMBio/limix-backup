import sys
import limix
from limix.core.covar import LowRankCov
from limix.core.covar import FixedCov
from limix.core.covar import FreeFormCov
from limix.core.gp import GP3KronSumLR
import scipy as sp
import scipy.stats as st
from limix.mtSet.core.iset_utils import * 
import numpy as np
import numpy.linalg as nla
import scipy.linalg as la
import copy
import pdb
from limix.utils.preprocess import gaussianize
from scipy.optimize import fmin

class ISet_Full():

    def __init__(self, Y=None, U_R=None, S_R=None, Xr=None, factr=1e7):
        """
        Args:
            Y:          [N, P] phenotype matrix
            S_R:        N vector of eigenvalues of R
            U_R:        [N, N] eigenvector matrix of R
            Xr:         [N, S] genotype data of the set component
            factr:      paramenter that determines the accuracy of the solution
                        (see scipy.optimize.fmin_l_bfgs_b for more details)
        """
        self.Y = Y
        self.U_R = U_R
        self.S_R = S_R
        self.XXh = U_R * S_R[sp.newaxis, :]**(0.5)
        self.Xr = Xr
        self.mtSet1 = limix.MTSet(Y=Y, S_R=S_R, U_R=U_R, rank=1)
        self.mtSet2 = limix.MTSet(Y=Y, S_R=S_R, U_R=U_R, rank=2)
        P = self.Y.shape[1]
        self.gp_block = GP3KronSumLR(Y=Y, Cr=FixedCov(sp.ones((P,P))), Cg=FreeFormCov(P),
                                     Cn=FreeFormCov(P), U_R=U_R, S_R=S_R, G=Xr)
        self.cache = {}
        self.factr = factr 

    def setXr(self, Xr):
        """ set genotype data of the set component """
        self.Xr = Xr
        self.gp_block.covar.G = Xr

    def fitNull(self, init_method='emp_cov'):
        """ fit null model """
        self.null = self.mtSet1.fitNull(cache=False, factr=self.factr, init_method=init_method)
        self.null['NLL'] = self.null['NLL0']
        self.mtSet2.null = copy.copy(self.null)
        return self.null

    def fitBlock(self, init_method='fr'):
        #print '.. fitting block model'
        if init_method=='fr':
            self.gp_block.covar.Cr.scale = self.fr['Cr'].mean()
            self.gp_block.covar.Cg.setCovariance(self.fr['Cg'])
            self.gp_block.covar.Cn.setCovariance(self.fr['Cn'])
        elif init_method=='null':
            self.gp_block.covar.Cr.scale = 1e-4
            self.gp_block.covar.Cg.setCovariance(self.null['Cg'])
            self.gp_block.covar.Cn.setCovariance(self.null['Cn'])
        elif init_method=='null_no_opt':
            self.gp_block.covar.Cr.scale = 1e-9
            self.gp_block.covar.Cg.setCovariance(self.null['Cg'])
            self.gp_block.covar.Cn.setCovariance(self.null['Cn'])
        conv, info = self.gp_block.optimize(factr=self.factr, verbose=False)

        self.block = {}
        # covars
        self.block['Cr'] = self.gp_block.covar.Cr.K() 
        self.block['Cg'] = self.gp_block.covar.Cg.K() 
        self.block['Cn'] = self.gp_block.covar.Cn.K() 
        self.block['NLL'] = sp.array([self.gp_block.LML()])
        self.block['LML_grad'] = self.gp_block.LML_grad()
        self.block['conv'] = sp.array([conv])

        # var comps
        self.block['var_r'] = sp.array([var_CoXX(self.block['Cr'], self.Xr)])
        self.block['var_g'] = sp.array([var_CoXX(self.block['Cg'], self.XXh)])
        self.block['var_n'] = sp.array([var_CoXX(self.block['Cn'], sp.eye(self.XXh.shape[0]))])

        return self.block

    def fitLowRank(self, init_method='fr'):
        assert self.Xr is not None, 'Set Xr!'
        #print '.. fitting lowrank model'
        if init_method=='fr': 
            self.mtSet1.Cr.setCovariance(self.fr['Cr'])
            self.mtSet1.Cg.setCovariance(self.fr['Cg'])
            self.mtSet1.Cn.setCovariance(self.fr['Cn'])
        elif init_method=='emp_cov':
            C = sp.cov(self.Y.T) / 3.
            self.mtSet1.Cr.setCovariance(C)
            self.mtSet1.Cg.setCovariance(C)
            self.mtSet1.Cn.setCovariance(C)
        else:
            X1 = sp.randn(2,2)
            X2 = sp.randn(2,2)
            X3 = sp.randn(2,2)
            C1 = sp.dot(X1, X1.T)
            C2 = sp.dot(X2, X2.T)
            C3 = sp.dot(X1, X1.T)
            C1 *= 1e-4 / C1.diagonal().mean()
            C2 *= .5 / C2.diagonal().mean()
            C3 *= .5 / C3.diagonal().mean()
            self.mtSet1.Cr.setCovariance(C1)
            self.mtSet1.Cg.setCovariance(C2)
            self.mtSet1.Cn.setCovariance(C3)
        params0 = self.mtSet1._gp.getParams()['covar']
        self.lr = self.mtSet1.optimize(self.Xr, params0=params0, factr=self.factr)
        self.lr['NLL'] = self.lr['NLLAlt']
        # var comps
        self.lr['var_r'] = sp.array([var_CoXX(self.lr['Cr'], self.Xr)])
        self.lr['var_g'] = sp.array([var_CoXX(self.lr['Cg'], self.XXh)])
        self.lr['var_n'] = sp.array([var_CoXX(self.lr['Cn'], sp.eye(self.XXh.shape[0]))])
        return self.lr

    def fitFullRank(self, init_method='emp_cov'):
        assert self.Xr is not None, 'Set Xr!'
        #print '.. fitting full model'
        if init_method=='emp_cov':
            C = sp.cov(self.Y.T) / 3.
            self.mtSet2.Cr.setCovariance(C)
            self.mtSet2.Cg.setCovariance(C)
            self.mtSet2.Cn.setCovariance(C)
        elif init_method=='lr':
            _Cr = self.lr['Cr'] + 1e-4 * sp.eye(self.lr['Cr'].shape[0])
            self.mtSet2.Cr.setCovariance(_Cr)
            self.mtSet2.Cg.setCovariance(self.lr['Cg'])
            self.mtSet2.Cn.setCovariance(self.lr['Cn'])
        else:
            X1 = sp.randn(2,2)
            X2 = sp.randn(2,2)
            X3 = sp.randn(2,2)
            C1 = sp.dot(X1, X1.T)
            C2 = sp.dot(X2, X2.T)
            C3 = sp.dot(X1, X1.T)
            C1 *= 1e-4 / C1.diagonal().mean()
            C2 *= .5 / C2.diagonal().mean()
            C3 *= .5  / C3.diagonal().mean()
            self.mtSet2.Cr.setCovariance(C1)
            self.mtSet2.Cg.setCovariance(C2)
            self.mtSet2.Cn.setCovariance(C3)
        params0 = self.mtSet2._gp.getParams()['covar']
        self.fr = self.mtSet2.optimize(self.Xr, params0=params0, factr=self.factr)
        self.fr['NLL'] = self.fr['NLLAlt']
        # var comps
        self.fr['var_r'] = sp.array([var_CoXX(self.fr['Cr'], self.Xr)])
        self.fr['var_g'] = sp.array([var_CoXX(self.fr['Cg'], self.XXh)])
        self.fr['var_n'] = sp.array([var_CoXX(self.fr['Cn'], sp.eye(self.XXh.shape[0]))])
        return self.fr

    def getVC(self):
        """
        Variance componenrs
        """
        _Cr = decompose_GxE(self.fr['Cr'])
        RV = {}
        for key in list(_Cr.keys()):
            RV['var_%s' % key] = sp.array([var_CoXX(_Cr[key], self.Xr)])
        RV['var_g'] = sp.array([var_CoXX(self.fr['Cg'], self.XXh)])
        RV['var_n'] = sp.array([var_CoXX(self.fr['Cn'], sp.eye(self.XXh.shape[0]))])
        return RV

    def _sim_from(self, set_covar='block', seed=None, qq=False):
        ##1. region term
        if set_covar=='block':
            Cr = self.block['Cr']
            Cg = self.block['Cg']
            Cn = self.block['Cn']
        if set_covar=='rank1':
            Cr = self.lr['Cr']
            Cg = self.lr['Cg']
            Cn = self.lr['Cn']
        Lc = msqrt(Cr)
        U, Sh, V = nla.svd(self.Xr, full_matrices=0)
        Lr = sp.zeros((self.Y.shape[0], self.Y.shape[0]))
        Lr[:, :Sh.shape[0]] = U * Sh[sp.newaxis, :]
        Z = sp.randn(*self.Y.shape)
        Yr = sp.dot(Lr, sp.dot(Z, Lc.T))
        ##2. bg term
        Lc = msqrt(Cg)
        Lr = self.XXh
        Z = sp.randn(*self.Y.shape)
        Yg = sp.dot(Lr, sp.dot(Z, Lc.T))
        # noise terms
        Lc = msqrt(Cn)
        Z = sp.randn(*self.Y.shape)
        Yn = sp.dot(Z, Lc.T)
        # normalize
        Y = Yr + Yg + Yn
        if qq:
            Y = gaussianize(Y)
            Y-= Y.mean(0)
            Y/= Y.std(0)
        return Y

