import sys
from limix.core.mean import MeanKronSum
from limix.core.covar import Cov2KronSumLR
from limix.core.covar import Covariance
from limix.core.type.cached import *

import pdb
import numpy as NP
import scipy as sp
import scipy.linalg as LA
import time as TIME
from limix.core.gp import GP
from limix.core.covar.cov_reml import cov_reml
from limix.utils.util_functions import vec
from limix.core.utils import assert_type_or_list_type
from limix.core.utils import assert_type
from limix.core.utils import assert_subtype


class GP2KronSumLR(GP):

    def __init__(self, Y, F, A, Cn, G, rank=1):
        """
        GP2KronSum specialized when the first covariance is lowrank
        Y:      Phenotype matrix
        Cn:     LIMIX trait-to-trait covariance for noise
        rank:   rank of the region term
        G:      Region term NxS (Remark: fast inference requires S<<N)
        """
        print 'pass XX and S_XX to covariance: the covariance should be responsable of caching stuff'

        assert_type(Y, NP.ndarray, 'Y')
        assert_type_or_list_type(F, NP.ndarray, 'F')
        assert_type_or_list_type(A, NP.ndarray, 'A')
        assert_subtype(Cn, Covariance, 'Cn')
        assert_type(G, NP.ndarray, 'G')

        covar = Cov2KronSumLR(Cn=Cn, G=G, rank=rank)
        covar.setRandomParams()
        mean = MeanKronSum(Y=Y, F=F, A=A)
        assert mean.n_terms == 1, ('GP2KronSum supports MeanKronSum'
                                   ' means with 1 term!')
        GP.__init__(self, covar=covar, mean=mean)

    def _observe(self):
        self.covar.register(self.col_cov_has_changed, 'row_cov')
        self.covar.register(self.col_cov_has_changed, 'col_cov')
        self.mean.register(self.pheno_has_changed, 'pheno')
        self.mean.register(self.designs_have_changed, 'designs')

    def _calc_all(self):
        # need computing of LML_grad otherwise default group is empty
        R = {}
        for key in self._cache_groups['default']:
            try:    R[key] = getattr(self, key)()
            except: R[key] = getattr(self, key)(0)
        return R

    def col_cov_has_changed_debug(self):
        # debug function for col_cov_has_changed_debug
        RV1  = self._calc_all()
        self.covar.setRandomParams()
        RV2  = self._calc_all()
        for key in list_keys:
            are_the_same = sp.array([((RV1[key]-RV2[key])**2)==0]).all()
            if are_the_same:   print key

    def col_cov_has_changed(self):
        self.clear_cache('col_cov')
        self.clear_all()


    def row_cov_has_changed(self):
        self.clear_cache('row_cov')
        self.clear_all()

    def pheno_has_changed(self):
        self.clear_cache('pheno')
        self.clear_all()

    def designs_have_changed(self):
        self.clear_cache('designs')
        self.clear_all()

    ######################
    # Transformed phenotype
    ######################
    @cached(['designs', 'pheno'])
    def FY(self):
        return self.mean.Ft_dot(self.mean.Y)

    @cached(['row_cov', 'pheno'])
    def WrY(self):
        return sp.dot(self.covar.Wr(), self.mean.Y)

    @cached(['row_cov', 'pheno'])
    def GGY(self):
        return sp.dot(self.covar.G, sp.dot(self.covar.G.T, self.mean.Y))

    @cached(['row_cov', 'pheno'])
    def WrGGY(self):
        return sp.dot(self.covar.Wr(), self.GGY())

    @cached(['col_cov', 'pheno'])
    def YLc(self):
        return sp.dot(self.mean.Y, self.covar.Lc().T)

    @cached(['row_col', 'col_cov', 'pheno'])
    def WrYLc(self):
        return sp.dot(self.WrY(), self.covar.Lc().T)

    @cached(['row_col', 'col_cov', 'pheno'])
    def WrYLcWc(self):
        return sp.dot(self.WrYLc(), self.covar.Wc().T)

    @cached(['designs', 'col_cov', 'pheno'])
    def FYLc(self):
        return sp.dot(self.FY(), self.covar.Lc().T)

    @cached(['designs', 'col_cov', 'pheno'])
    def FYLcALc(self):
        return sp.dot(self.FYLc(), self.ALc().T)

    @cached(['row_col', 'col_cov', 'pheno'])
    def GGYLc(self):
        return sp.dot(self.GGY(), self.covar.Lc().T)

    @cached(['row_col', 'col_cov', 'pheno'])
    def WrGGYLc(self):
        return sp.dot(self.WrGGY(), self.covar.Lc().T)

    @cached(['row_col', 'col_cov', 'pheno'])
    def DWrYLcWc(self):
        return self.covar.D() * self.WrYLcWc()

    @cached(['row_col', 'col_cov', 'pheno'])
    def RYLcCtilde(self, i):
        if i < self.covar.Cg.getNumberParams():
            RYLc = self.GGYLc()
        else:
            RYLc = self.YLc()
        return sp.dot(RYLc, self.covar.Ctilde(i))

    @cached(['row_col', 'col_cov', 'pheno'])
    def SrDWrYLcWcCbar(self, i):
        if i < self.covar.Cg.getNumberParams():
            SgDWrYLcWc = self.covar.Sg()[:, sp.newaxis] *  self.DWrYLcWc()
        else:
            SgDWrYLcWc = self.DWrYLcWc()
        return sp.dot(SgDWrYLcWc, self.covar.Cbar(i))

    @cached(['row_col', 'col_cov', 'pheno'])
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
    @cached(['designs', 'row_col'])
    def WrF(self):
        return sp.dot(self.covar.Wr(), self.mean.F[0])

    @cached(['designs', 'row_col'])
    def GF(self):
        return sp.dot(self.covar.G.T, self.mean.F[0])

    @cached(['designs', 'row_col'])
    def GGF(self):
        return sp.dot(self.covar.G, self.GF())

    @cached(['designs', 'row_col'])
    def FGGF(self):
        return sp.dot(self.GF().T, self.GF())

    @cached('designs')
    def FF(self):
        return self.mean.Ft_dot(self.mean.F[0])

    @cached(['designs', 'row_col'])
    def WrGGF(self):
        return sp.dot(self.covar.Wr(), self.GGF())

    @cached(['designs', 'row_col', 'col_cov', 'pheno'])
    def FB(self):
        return sp.dot(self.mean.F[0], self.mean.B[0])

    @cached(['designs', 'row_col', 'col_cov', 'pheno'])
    def WrFB(self):
        return sp.dot(self.WrF(), self.mean.B[0])

    @cached(['designs', 'row_col', 'col_cov', 'pheno'])
    def GGFB(self):
        return sp.dot(self.GGF(), self.mean.B[0])

    @cached(['designs', 'row_col', 'col_cov', 'pheno'])
    def WrGGFB(self):
        return sp.dot(self.WrGGF(), self.mean.B[0])

    @cached(['designs', 'row_col', 'col_cov', 'pheno'])
    def FBALc(self):
        return sp.dot(self.FB(), self.ALc())

    @cached(['designs', 'row_col', 'col_cov', 'pheno'])
    def WrFBALc(self):
        return sp.dot(self.WrFB(), self.ALc())

    @cached(['designs', 'row_col', 'col_cov', 'pheno'])
    def GGFBALc(self):
        return sp.dot(self.GGFB(), self.ALc())

    @cached(['designs', 'row_col', 'col_cov', 'pheno'])
    def WrGGFBALc(self):
        return sp.dot(self.WrGGFB(), self.ALc())

    @cached(['designs', 'row_col', 'col_cov', 'pheno'])
    def WrGGFBALcWc(self):
        return sp.dot(self.WrGGFBALc(), self.covar.Wc().T)

    @cached(['designs', 'row_col', 'col_cov', 'pheno'])
    def WrFBALcWc(self):
        return sp.dot(self.WrFBALc(), self.covar.Wc().T)

    @cached(['designs', 'row_col', 'col_cov', 'pheno'])
    def DWrGGFBALcWc(self):
        return self.covar.D() * self.WrGGFBALcWc()

    @cached(['designs', 'row_col', 'col_cov', 'pheno'])
    def DWrFBALcWc(self):
        return self.covar.D() * self.WrFBALcWc()

    @cached(['designs', 'col_cov'])
    def ALc(self):
        return sp.dot(self.mean.A[0], self.covar.Lc().T)

    @cached(['designs', 'col_cov'])
    def ALcLcA(self):
        return sp.dot(self.ALc(), self.ALc().T)

    @cached(['designs', 'col_cov'])
    def ALcWc(self):
        return sp.dot(self.ALc(), self.covar.Wc().T)

    @cached(['designs', 'row_col', 'col_cov'])
    def WLW(self):
        # confusing notation the first W refers to kroneckered matrix
        # that brings to the low-dimensional space while the second refers to the fixed effect design
        return sp.kron(self.ALcWc().T, self.WrF())

    @cached(['designs', 'row_col', 'col_cov'])
    def dWLW(self):
        return self.covar.d()[:,sp.newaxis] * self.WLW()

    @cached(['designs', 'row_col', 'col_cov', 'pheno'])
    def RFBALcCtilde(self, i):
        if i < self.covar.Cg.getNumberParams():
            RFBALc = self.GGFBALc()
        else:
            RFBALc = self.FBALc()
        return sp.dot(RFBALc, self.covar.Ctilde(i))

    @cached(['designs', 'row_col', 'col_cov', 'pheno'])
    def SrDWrFBALcWcCbar(self, i):
        if i < self.covar.Cg.getNumberParams():
            SgDWrFBALcWc = self.covar.Sg()[:, sp.newaxis] *  self.DWrFBALcWc()
        else:
            SgDWrFBALcWc = self.DWrFBALcWc()
        return sp.dot(SgDWrFBALcWc, self.covar.Cbar(i))

    @cached(['designs', 'row_col', 'col_cov', 'pheno'])
    def WrRFBALcCtildeWc(self, i):
        if i < self.covar.Cg.getNumberParams():
            WrRFBALc = self.WrGGFBALc()
        else:
            WrRFBALc = self.WrFBALc()
        return sp.dot(WrRFBALc, sp.dot(self.covar.Ctilde(i), self.covar.Wc().T))

    # fixed effect methods for Areml_grad

    @cached('col_cov')
    def CtildeLcA(self, i):
        return sp.dot(self.covar.Ctilde(i), self.ALc().T)

    @cached(['designs', 'row_col', 'col_cov'])
    def ALcCtildeLcA_o_FRF(self, i):
        if i < self.covar.Cg.getNumberParams():
            FRF = self.FGGF()
        else:
            FRF = self.FF()
        ALcCtildeALc = sp.dot(self.ALc(), self.CtildeLcA(i))
        return sp.kron(ALcCtildeALc, FRF)

    @cached(['designs', 'row_col', 'col_cov'])
    def Cbar_o_Sr_dWLW(self, i):
        if i < self.covar.Cg.getNumberParams():
            Cbar_o_Sr = sp.kron(self.covar.Cbar(i), sp.diag(self.covar.Sg()))
        else:
            Cbar_o_Sr = sp.kron(self.covar.Cbar(i), sp.eye(self.covar.rank_r))
        return sp.dot(Cbar_o_Sr, self.dWLW())

    @cached(['designs', 'row_col', 'col_cov'])
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
    @cached(['designs', 'row_col', 'col_cov', 'pheno'])
    def WKiy(self):
        R = vec(self.FYLcALc())
        # the following could be further optimized but do not care for now
        R -= sp.dot(self.WLW().T, vec(self.DWrYLcWc()))
        return R

    def update_b(self):
        if self.mean.n_covs>0:
            self.mean.b = self.Areml.solve(self.WKiy())

    @cached(['row_col', 'col_cov', 'pheno'])
    def yKiy(self):
        return (self.YLc() * self.YLc()).sum() - (self.WrYLcWc() * self.DWrYLcWc()).sum()

    @cached(['designs', 'row_col', 'col_cov', 'pheno'])
    def yKiWb(self):
        return (self.WKiy() * self.mean.b).sum()

    #########################
    # Gradients
    #########################
    @cached(['row_col', 'col_cov', 'pheno'])
    def yKiy_grad_i(self,i):
        r = - (self.YLc() * self.RYLcCtilde(i)).sum()
        r+= - (self.DWrYLcWc() * self.SrDWrYLcWcCbar(i)).sum()
        r+= 2 * (self.WrRYLcCtildeWc(i) * self.DWrYLcWc()).sum()
        return r

    @cached(['designs', 'row_col', 'col_cov', 'pheno'])
    def yKiWb_grad_i(self,i):
        r = -2 * (self.YLc() * self.RFBALcCtilde(i)).sum()
        r+= -2 * (self.DWrYLcWc() * self.SrDWrFBALcWcCbar(i)).sum()
        r+= 2 * (self.WrRYLcCtildeWc(i) * self.DWrFBALcWc()).sum()
        r+= 2 * (self.WrRFBALcCtildeWc(i) * self.DWrYLcWc()).sum()
        r+= (self.FBALc() * self.RFBALcCtilde(i)).sum()
        r+= (self.DWrFBALcWc() * self.SrDWrFBALcWcCbar(i)).sum()
        r+= - 2 * (self.WrRFBALcCtildeWc(i) * self.DWrFBALcWc()).sum()
        return r
