import sys
from limix.core.mean import MeanKronSum
from limix.core.covar import Cov2KronSumLR 
from limix.core.type.cached import *

import pdb
import numpy as NP
import scipy as sp
import scipy.linalg as LA
import time as TIME
from limix.core.gp import GP
from limix.core.covar.cov_reml import cov_reml
from limix.utils.util_functions import vec

class GP2KronSumLR(GP):

    def __init__(self,Y = None, F = None, A = None, Cn = None, G = None, rank = 1):
        """
        GP2KronSum specialized when the first covariance is lowrank 
        Y:      Phenotype matrix
        Cn:     LIMIX trait-to-trait covariance for noise
        rank:   rank of the region term
        G:      Region term NxS (Remark: fast inference requires S<<N)
        """
        print 'pass XX and S_XX to covariance: the covariance should be responsable of caching stuff'
        self.covar = Cov2KronSumLR(Cn = Cn, G = G, rank = rank)
        print 'delete the following line when the caching will work'
        self.covar.setRandomParams()
        self.mean  = MeanKronSum(Y = Y, F = F, A = A)
        assert self.mean.n_terms==1, 'GP2KronSum supports MeanKronSum means with 1 term!'
        self.Areml = cov_reml(self)
        self.update_b()
        self.covar.register(self.col_cov_has_changed, 'row_cov')
        self.covar.register(self.col_cov_has_changed, 'col_cov')
        self.mean.register(self.pheno_has_changed, 'pheno')
        self.mean.register(self.designs_have_changed, 'designs')

    # LISTING THEM ALL (easier removing than adding)
    #also, the ones that are not used more than one time can be removed
    #self.clear_cache('FY','WrY','GGY','WrGGY',
    #'YLc','WrYLc','WrYLcWc','FYLc','FYLcALc','GGYLc','WrGGYLc',
    #'DWrYLcWc','RYLcCtilde','SrDWrYLcWcCbar','WrRYLcCtildeWc',
    #'WrF','GF','GGF','FGGF','FF','WrGGF','FB','WrFB','GGFB',
    #'WrGGFB','FBALc','WrFBALc','GGFBALc','WrGGFBALc',
    #'WrGGFBALcWc','WrFBALcWc','DWrGGFBALcWc','DWrFBALcWc',
    #'ALc','ALcLcA','ALcWc','WLW','dWLW','RFBALcCtilde',
    #'SrDWrFBALcWcCbar','WrRFBALcCtildeWc','CtildeLcA',
    #'ALcCtildeLcA_o_FRF','Cbar_o_Sr_dWLW','WcCtildeLcA_o_WrRF',
    #'WKiy', 'yKiy', 'yKiWb', 'yKiy_grad_i','yKiWb_grad_i')

    def col_cov_has_changed(self):
        self.clear_cache('YLc','WrYLc','WrYLcWc', 'FYLc', 'FYLcALc',
                            'GGYLc', 'WrGGYLc', 'DWrYLcWc',
                            'RYLcCtilde', 'SrDWrYLcWcCbar',
                            'WrRYLcCtildeWc','FB','WrFB','GGFB',
                            'WrGGFB', 'FBALc', 'WrFBALc',
                            'GGFBALc', 'WrGGFBALc', 'WrGGFBALcWc',
                            'WrFBALcWc', 'DWrGGFBALcWc',
                            'DWrFBALcWc', 'ALc', 'ALcLcA',
                            'ALcWc', 'WLW', 'dWLW', 'RFBALcCtilde',
                            'SrDWrFBALcWcCbar', 'WrRFBALcCtildeWc',
                            'CtildeLcA', 'ALcCtildeLcA_o_FRF',
                            'Cbar_o_Sr_dWLW', 'WcCtildeLcA_o_WrRF',
                            'WKiy', 'yKiy', 'yKiWb', 'yKiy_grad_i',
                            'yKiWb_grad_i')
        self.clear_all()

    def col_cov_has_changed_debug(self):
        list_keys = ['FY','WrY','GGY','WrGGY',
                        'YLc','WrYLc','WrYLcWc','FYLc','FYLcALc','GGYLc','WrGGYLc',
                        'DWrYLcWc','RYLcCtilde','SrDWrYLcWcCbar','WrRYLcCtildeWc',
                        'WrF','GF','GGF','FGGF','FF','WrGGF','FB','WrFB','GGFB',
                        'WrGGFB','FBALc','WrFBALc','GGFBALc','WrGGFBALc',
                        'WrGGFBALcWc','WrFBALcWc','DWrGGFBALcWc','DWrFBALcWc',
                        'ALc','ALcLcA','ALcWc','WLW','dWLW','RFBALcCtilde',
                        'SrDWrFBALcWcCbar','WrRFBALcCtildeWc','CtildeLcA',
                        'ALcCtildeLcA_o_FRF','Cbar_o_Sr_dWLW','WcCtildeLcA_o_WrRF',
                        'WKiy', 'yKiy', 'yKiWb', 'yKiy_grad_i','yKiWb_grad_i']

        list_keys_i = ['RYLcCtilde', 'SrDWrYLcWcCbar','WrRYLcCtildeWc',
                            'RFBALcCtilde','SrDWrFBALcWcCbar', 'WrRFBALcCtildeWc',
                            'CtildeLcA', 'ALcCtildeLcA_o_FRF',
                            'Cbar_o_Sr_dWLW', 'WcCtildeLcA_o_WrRF',
                            'yKiy_grad_i','yKiWb_grad_i']
        pdb.set_trace()
        RV1  = {}
        for key in list_keys:
            if key in list_keys_i:
                RV1[key] = getattr(self, key)(0)
            else:
                RV1[key] = getattr(self, key)()
        self.covar.setRandomParams()
        RV2 = {}
        for key in list_keys:
            if key in list_keys_i:
                RV2[key] = getattr(self, key)(0)
            else:
                RV2[key] = getattr(self, key)()
        for key in list_keys:
            r = sp.array([((RV1[key]-RV2[key])**2)==0]).all()
            if r:   print key
        pdb.set_trace()
        

    def row_cov_has_changed(self):
        self.clear_cache('WrY', 'GGY', 'WrGGY', 'WrYLc',
        'WrYLcWc','GGYLc','WrGGYLc','DWrYLcWc','RYLcCtilde',
        'SrDWrYLcWcCbar','WrRYLcCtildeWc','WrF','GF','GGF',
        'FGGF','WrGGF','FB','WrFB','GGFB','WrGGFB','FBALc',
        'WrFBALc','GGFBALc','WrGGFBALc','WrGGFBALcWc',
        'WrFBALcWc','DWrGGFBALcWc','DWrFBALcWc','WLW','dWLW',
        'RFBALcCtilde','SrDWrFBALcWcCbar','WrRFBALcCtildeWc',
        'ALcCtildeLcA_o_FRF','Cbar_o_Sr_dWLW','WcCtildeLcA_o_WrRF',
        'WKiy', 'yKiy', 'yKiWb', 'yKiy_grad_i','yKiWb_grad_i')
        self.clear_all()

    def pheno_has_changed(self):
        self.clear_cache('FY','WrY','GGY','WrGGY',
        'YLc','WrYLc','WrYLcWc','FYLc','FYLcALc','GGYLc','WrGGYLc',
        'DWrYLcWc','RYLcCtilde','SrDWrYLcWcCbar','WrRYLcCtildeWc',
        'FB','WrFB','GGFB','WrGGFB','FBALc',
        'WrFBALc','GGFBALc','WrGGFBALc',
        'WrGGFBALcWc','WrFBALcWc','DWrGGFBALcWc','DWrFBALcWc',
        'RFBALcCtilde','SrDWrFBALcWcCbar','WrRFBALcCtildeWc',
        'WKiy', 'yKiy', 'yKiWb', 'yKiy_grad_i','yKiWb_grad_i')
        self.clear_all()

    def designs_have_changed(self):
        self.clear_cache('FY','FYLc','FYLcALc',
        'WrF','GF','GGF','FGGF','FF','WrGGF','FB','WrFB','GGFB',
        'WrGGFB','FBALc','WrFBALc','GGFBALc','WrGGFBALc',
        'WrGGFBALcWc','WrFBALcWc','DWrGGFBALcWc','DWrFBALcWc',
        'ALc','ALcLcA','ALcWc','WLW','dWLW','RFBALcCtilde',
        'SrDWrFBALcWcCbar','WrRFBALcCtildeWc','CtildeLcA',
        'ALcCtildeLcA_o_FRF','Cbar_o_Sr_dWLW','WcCtildeLcA_o_WrRF',
        'WKiy', 'yKiy', 'yKiWb', 'yKiy_grad_i','yKiWb_grad_i')
        self.clear_all()

    ######################
    # Transformed phenotype
    ######################
    @cached
    def FY(self):
        return self.mean.Ft_dot(self.mean.Y) 

    @cached
    def WrY(self):
        return sp.dot(self.covar.Wr(), self.mean.Y)

    @cached
    def GGY(self):
        return sp.dot(self.covar.G, sp.dot(self.covar.G.T, self.mean.Y))

    @cached
    def WrGGY(self):
        return sp.dot(self.covar.Wr(), self.GGY()) 

    @cached
    def YLc(self):
        return sp.dot(self.mean.Y, self.covar.Lc().T)

    @cached
    def WrYLc(self):
        return sp.dot(self.WrY(), self.covar.Lc().T)

    @cached
    def WrYLcWc(self):
        return sp.dot(self.WrYLc(), self.covar.Wc().T)

    @cached
    def FYLc(self):
        return sp.dot(self.FY(), self.covar.Lc().T)

    @cached
    def FYLcALc(self):
        return sp.dot(self.FYLc(), self.ALc().T)

    @cached
    def GGYLc(self):
        return sp.dot(self.GGY(), self.covar.Lc().T) 

    @cached
    def WrGGYLc(self):
        return sp.dot(self.WrGGY(), self.covar.Lc().T)

    @cached
    def DWrYLcWc(self):
        return self.covar.D() * self.WrYLcWc() 

    @cached
    def RYLcCtilde(self, i):
        if i < self.covar.Cg.getNumberParams():
            RYLc = self.GGYLc()
        else:
            RYLc = self.YLc()
        return sp.dot(RYLc, self.covar.Ctilde(i))

    @cached
    def SrDWrYLcWcCbar(self, i):
        if i < self.covar.Cg.getNumberParams():
            SgDWrYLcWc = self.covar.Sg()[:, sp.newaxis] *  self.DWrYLcWc() 
        else:
            SgDWrYLcWc = self.DWrYLcWc()
        return sp.dot(SgDWrYLcWc, self.covar.Cbar(i))

    @cached
    def WrRYLcCtildeWc(self, i):
        if i < self.covar.Cg.getNumberParams():
            WrRYLc = self.WrGGYLc() 
        else:
            WrRYLc = self.WrYLc() 
        return sp.dot(WrRYLc, sp.dot(self.covar.Ctilde(i), self.covar.Wc().T))

    ###############################################
    # Fixed effects
    # uses that MeanKronSum has only one term
    ###############################################
    @cached
    def WrF(self):
        return sp.dot(self.covar.Wr(), self.mean.F[0])

    @cached
    def GF(self):
        return sp.dot(self.covar.G.T, self.mean.F[0])

    @cached
    def GGF(self):
        return sp.dot(self.covar.G, self.GF()) 

    @cached
    def FGGF(self):
        return sp.dot(self.GF().T, self.GF())

    @cached
    def FF(self):
        return self.mean.Ft_dot(self.mean.F[0])

    @cached
    def WrGGF(self):
        return sp.dot(self.covar.Wr(), self.GGF())

    @cached
    def FB(self):
        return sp.dot(self.mean.F[0], self.mean.B[0])

    @cached
    def WrFB(self):
        return sp.dot(self.WrF(), self.mean.B[0])

    @cached
    def GGFB(self):
        return sp.dot(self.GGF(), self.mean.B[0])

    @cached
    def WrGGFB(self):
        return sp.dot(self.WrGGF(), self.mean.B[0])

    @cached
    def FBALc(self):
        return sp.dot(self.FB(), self.ALc())

    @cached
    def WrFBALc(self):
        return sp.dot(self.WrFB(), self.ALc())

    @cached
    def GGFBALc(self):
        return sp.dot(self.GGFB(), self.ALc())

    @cached
    def WrGGFBALc(self):
        return sp.dot(self.WrGGFB(), self.ALc())

    @cached
    def WrGGFBALcWc(self):
        return sp.dot(self.WrGGFBALc(), self.covar.Wc().T)

    @cached
    def WrFBALcWc(self):
        return sp.dot(self.WrFBALc(), self.covar.Wc().T)

    @cached
    def DWrGGFBALcWc(self):
        return self.covar.D() * self.WrGGFBALcWc()

    @cached
    def DWrFBALcWc(self):
        return self.covar.D() * self.WrFBALcWc()

    @cached
    def ALc(self):
        return sp.dot(self.mean.A[0], self.covar.Lc().T)

    @cached
    def ALcLcA(self):
        return sp.dot(self.ALc(), self.ALc().T)

    @cached
    def ALcWc(self):
        return sp.dot(self.ALc(), self.covar.Wc().T)

    @cached
    def WLW(self):
        # confusing notation the first W refers to kroneckered matrix
        # that brings to the low-dimensional space while the second refers to the weights
        return sp.kron(self.ALcWc().T, self.WrF()) 

    @cached
    def dWLW(self):
        return self.covar.d()[:,sp.newaxis] * self.WLW()

    @cached
    def RFBALcCtilde(self, i):
        if i < self.covar.Cg.getNumberParams():
            RFBALc = self.GGFBALc()
        else:
            RFBALc = self.FBALc()
        return sp.dot(RFBALc, self.covar.Ctilde(i))

    @cached
    def SrDWrFBALcWcCbar(self, i):
        if i < self.covar.Cg.getNumberParams():
            SgDWrFBALcWc = self.covar.Sg()[:, sp.newaxis] *  self.DWrFBALcWc()
        else:
            SgDWrFBALcWc = self.DWrFBALcWc()
        return sp.dot(SgDWrFBALcWc, self.covar.Cbar(i))

    @cached
    def WrRFBALcCtildeWc(self, i):
        if i < self.covar.Cg.getNumberParams():
            WrRFBALc = self.WrGGFBALc()
        else:
            WrRFBALc = self.WrFBALc()
        return sp.dot(WrRFBALc, sp.dot(self.covar.Ctilde(i), self.covar.Wc().T))

    # fixed effect methods for Areml_grad
    
    @cached
    def CtildeLcA(self, i):
        return sp.dot(self.covar.Ctilde(i), self.ALc().T)

    @cached
    def ALcCtildeLcA_o_FRF(self, i): 
        if i < self.covar.Cg.getNumberParams():
            FRF = self.FGGF()
        else:
            FRF = self.FF()
        ALcCtildeALc = sp.dot(self.ALc(), self.CtildeLcA(i))
        return sp.kron(ALcCtildeALc, FRF)

    @cached
    def Cbar_o_Sr_dWLW(self, i):
        if i < self.covar.Cg.getNumberParams():
            Cbar_o_Sr = sp.kron(self.covar.Cbar(i), sp.diag(self.covar.Sg())) 
        else:
            Cbar_o_Sr = sp.kron(self.covar.Cbar(i), sp.eye(self.covar.rank_r)) 
        return sp.dot(Cbar_o_Sr, self.dWLW())

    @cached
    def WcCtildeLcA_o_WrRF(self, i):
        if i < self.covar.Cg.getNumberParams():
            self.WrRF = self.WrGGF()
        else:
            self.WrRF = self.WrF()
        WcCtildeLcA = sp.dot(self.covar.Wc(), self.CtildeLcA(i))
        return sp.kron(WcCtildeLcA, self.WrRF)


    ######################
    # Areml
    ######################
    def Areml_K(self):
        R  = sp.kron(self.ALcLcA(), self.FF())
        R -= sp.dot(self.WLW().T, self.dWLW())
        return R

    def Areml_K_grad_i(self,i):
        R = sp.dot(self.WcCtildeLcA_o_WrRF(i).T, self.dWLW())
        R+= R.T
        R+= -self.ALcCtildeLcA_o_FRF(i)
        R+= -sp.dot(self.dWLW().T, self.Cbar_o_Sr_dWLW(i))
        return R

    ########################
    # LML terms
    ########################
    @cached
    def WKiy(self):
        R = vec(self.FYLcALc())
        # the following could be further optimized but do not care for now
        R -= sp.dot(self.WLW().T, vec(self.DWrYLcWc()))
        return R

    def update_b(self):
        self.mean.b = self.Areml.solve(self.WKiy())

    @cached
    def yKiy(self):
        return (self.YLc() * self.YLc()).sum() - (self.WrYLcWc() * self.DWrYLcWc()).sum()

    @cached
    def yKiWb(self):
        return (self.WKiy() * self.mean.b).sum()

    #########################
    # Gradients
    #########################
    @cached
    def yKiy_grad_i(self,i):
        r = - (self.YLc() * self.RYLcCtilde(i)).sum()
        r+= - (self.DWrYLcWc() * self.SrDWrYLcWcCbar(i)).sum()
        r+= 2 * (self.WrRYLcCtildeWc(i) * self.DWrYLcWc()).sum()
        return r 

    @cached
    def yKiWb_grad_i(self,i):
        r = -2 * (self.YLc() * self.RFBALcCtilde(i)).sum()
        r+= -2 * (self.DWrYLcWc() * self.SrDWrFBALcWcCbar(i)).sum()
        r+= 2 * (self.WrRYLcCtildeWc(i) * self.DWrFBALcWc()).sum()
        r+= 2 * (self.WrRFBALcCtildeWc(i) * self.DWrYLcWc()).sum()
        r+= (self.FBALc() * self.RFBALcCtilde(i)).sum()
        r+= (self.DWrFBALcWc() * self.SrDWrFBALcWcCbar(i)).sum()
        r+= - 2 * (self.WrRFBALcCtildeWc(i) * self.DWrFBALcWc()).sum()
        return r 


