import sys
from limix.core.mean import MeanKronSum
from limix.core.covar import Cov2KronSumLR
from limix.core.covar import Covariance
from hcache import Cached, cached

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
    """
    Gaussian Process with a 2kronSumLR Covariance and a mean that is a sum of Kronecker products:
        vec(Y) ~ N( vec( \sum_i F_i B_i A_i), Cr \kron GG.T + Cn \kron I )
    The current implementation supports only 1 fixed effect term!
    Notation:
        N = number of samples
        P = number of traits
        Y = [N, P] phenotype matrix
        F_i = sample fixed effect design for term i
        A_i = trait fixed effect design for term i
        B_i = effect sizes of fixed effect term i
        Cr = column covariance matrix for low-rank term
        Cn = column covariance matrix for noise term
        rank_c = rank of low-rank col covariance
        rank_r = rank of low-rank row covariance
    """

    def __init__(self, Y, Cn, G, F=None, A=None, rank=1):
        """
        Args:
            Y:      [N, P] phenotype matrix
            F:      Sample fixed effect design (first dimension must be N)
            A:      Trait fixed effect design (second dimension must be P)
            Cn:     Limix covariance matrix for Cn (dimension P)
            G:      [N, rank_r] numpy covariance matrix for G
            rank:   rank of column low-rank covariance (default = 1)
        """
        assert_type(Y, NP.ndarray, 'Y')
        assert_subtype(Cn, Covariance, 'Cn')
        assert_type(G, NP.ndarray, 'G')

        covar = Cov2KronSumLR(Cn=Cn, G=G, rank=rank)
        covar.setRandomParams()
        mean = MeanKronSum(Y=Y, F=F, A=A)
        assert mean.n_terms <= 1, ('GP2KronSum supports MeanKronSum'
                                   ' means with maximum 1 term!')
        GP.__init__(self, covar=covar, mean=mean)

    def _observe(self):
        self.covar.register(self.row_cov_has_changed, 'row_cov')
        self.covar.register(self.col_cov_has_changed, 'col_cov')
        self.mean.register(self.pheno_has_changed, 'pheno')
        self.mean.register(self.designs_have_changed, 'designs')

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
    @cached(['pheno'])
    def YY(self):
        return sp.dot(self.mean.Y.T, self.mean.Y)

    @cached(['designs', 'pheno'])
    def FY(self):
        return sp.dot(self.mean.F[0].T, self.mean.Y)

    @cached(['row_cov', 'pheno'])
    def WrY(self):
        return sp.dot(self.covar.Wr(), self.mean.Y)

    @cached(['row_cov', 'pheno'])
    def GGY(self):
        return sp.dot(self.covar.G, sp.dot(self.covar.G.T, self.mean.Y))

    @cached(['row_cov', 'pheno'])
    def YGGY(self):
        return sp.dot(self.mean.Y.T, self.GGY()) 

    @cached(['row_cov', 'pheno'])
    def WrGGY(self):
        return sp.dot(self.covar.Wr(), self.GGY())

    @cached(['col_cov', 'pheno'])
    def YLc(self):
        return sp.dot(self.mean.Y, self.covar.Lc().T)

    @cached(['row_cov', 'col_cov', 'pheno'])
    def WrYLc(self):
        return sp.dot(self.WrY(), self.covar.Lc().T)

    @cached(['row_cov', 'col_cov', 'pheno'])
    def WrYLcWc(self):
        return sp.dot(self.WrYLc(), self.covar.Wc().T)

    @cached(['designs', 'col_cov', 'pheno'])
    def FYLc(self):
        return sp.dot(self.FY(), self.covar.Lc().T)

    @cached(['designs', 'col_cov', 'pheno'])
    def FYLcALc(self):
        return sp.dot(self.FYLc(), self.ALc().T)

    @cached(['row_cov', 'col_cov', 'pheno'])
    def GGYLc(self):
        return sp.dot(self.GGY(), self.covar.Lc().T)

    @cached(['row_cov', 'col_cov', 'pheno'])
    def WrGGYLc(self):
        return sp.dot(self.WrGGY(), self.covar.Lc().T)

    @cached(['row_cov', 'col_cov', 'pheno'])
    def DWrYLcWc(self):
        return self.covar.D() * self.WrYLcWc()

    #@cached(['row_cov', 'col_cov', 'pheno'])
    #def RYLcCtilde(self, i):
    #    if i < self.covar.Cr.getNumberParams():
    #        RYLc = self.GGYLc()
    #    else:
    #        RYLc = self.YLc()
    #    return sp.dot(RYLc, self.covar.Ctilde(i))

    @cached(['row_cov', 'col_cov', 'pheno'])
    def SrDWrYLcWcCbar(self, i):
        if i < self.covar.Cr.getNumberParams():
            SgDWrYLcWc = self.covar.Sg()[:, sp.newaxis] *  self.DWrYLcWc()
        else:
            SgDWrYLcWc = self.DWrYLcWc()
        return sp.dot(SgDWrYLcWc, self.covar.Cbar(i))

    def YRY(self, i):
        if i < self.covar.Cr.getNumberParams():
            return self.YGGY()
        else:
            return self.YY()

    @cached(['row_cov', 'col_cov', 'pheno'])
    def WrRYLcCtildeWc(self, i):
        if i < self.covar.Cr.getNumberParams():
            WrRYLc = self.WrGGYLc()
        else:
            WrRYLc = self.WrYLc()
        return sp.dot(WrRYLc, sp.dot(self.covar.Ctilde(i), self.covar.Wc().T))

    ###############################################
    # Fixed effects
    # uses that MeanKronSum has only one term
    ###############################################
    @cached(['designs', 'row_cov'])
    def WrF(self):
        return sp.dot(self.covar.Wr(), self.mean.F[0])

    @cached(['designs', 'row_cov'])
    def GGF(self):
        return sp.dot(self.covar.G, sp.dot(self.covar.G.T, self.mean.F[0]))

    @cached(['designs', 'row_cov'])
    def YGGF(self):
        return sp.dot(self.mean.Y.T, self.GGF()) 

    @cached(['designs', 'row_cov'])
    def FGGF(self):
        return sp.dot(self.mean.F[0].T, self.GGF())

    @cached('designs')
    def FF(self):
        return sp.dot(self.mean.F[0].T, self.mean.F[0])

    @cached(['designs', 'row_cov'])
    def WrGGF(self):
        return sp.dot(self.covar.Wr(), self.GGF())

    @cached(['designs', 'row_cov', 'col_cov', 'pheno'])
    def FB(self):
        return sp.dot(self.mean.F[0], self.mean.B[0])

    @cached(['designs', 'row_cov', 'col_cov', 'pheno'])
    def YGGFBA(self):
        return sp.dot(sp.dot(self.YGGF(), self.mean.B[0]), self.mean.A[0]) 

    @cached(['designs', 'row_cov', 'col_cov', 'pheno'])
    def YFBA(self):
        return sp.dot(sp.dot(self.FY().T, self.mean.B[0]), self.mean.A[0]) 

    @cached(['designs', 'row_cov', 'col_cov', 'pheno'])
    def WrFB(self):
        return sp.dot(self.WrF(), self.mean.B[0])

    @cached(['designs', 'row_cov', 'col_cov', 'pheno'])
    def GGFB(self):
        return sp.dot(self.GGF(), self.mean.B[0])

    @cached(['designs', 'row_cov', 'col_cov', 'pheno'])
    def WrGGFB(self):
        return sp.dot(self.WrGGF(), self.mean.B[0])

    @cached(['designs', 'row_cov', 'col_cov', 'pheno'])
    def FBALc(self):
        return sp.dot(self.FB(), self.ALc())

    @cached(['designs', 'row_cov', 'col_cov', 'pheno'])
    def WrFBALc(self):
        return sp.dot(self.WrFB(), self.ALc())

    @cached(['designs', 'row_cov', 'col_cov', 'pheno'])
    def GGFBALc(self):
        return sp.dot(self.GGFB(), self.ALc())

    @cached(['designs', 'row_cov', 'col_cov', 'pheno'])
    def WrGGFBALc(self):
        return sp.dot(self.WrGGFB(), self.ALc())

    @cached(['designs', 'row_cov', 'col_cov', 'pheno'])
    def WrGGFBALcWc(self):
        return sp.dot(self.WrGGFBALc(), self.covar.Wc().T)

    @cached(['designs', 'row_cov', 'col_cov', 'pheno'])
    def WrFBALcWc(self):
        return sp.dot(self.WrFBALc(), self.covar.Wc().T)

    @cached(['designs', 'row_cov', 'col_cov', 'pheno'])
    def DWrGGFBALcWc(self):
        return self.covar.D() * self.WrGGFBALcWc()

    @cached(['designs', 'row_cov', 'col_cov', 'pheno'])
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

    @cached(['designs', 'row_cov', 'col_cov'])
    def WLW(self):
        # confusing notation the first W refers to kroneckered matrix
        # that brings to the low-dimensional space while the second refers to the fixed effect design
        return sp.kron(self.ALcWc().T, self.WrF())

    @cached(['designs', 'row_cov', 'col_cov'])
    def dWLW(self):
        return self.covar.d()[:,sp.newaxis] * self.WLW()

    #@cached(['designs', 'row_cov', 'col_cov', 'pheno'])
    #def RFBALcCtilde(self, i):
    #    if i < self.covar.Cr.getNumberParams():
    #        RFBALc = self.GGFBALc()
    #    else:
    #        RFBALc = self.FBALc()
    #    return sp.dot(RFBALc, self.covar.Ctilde(i))

    def YRFBA(self, i):
        if i < self.covar.Cr.getNumberParams():
            return self.YGGFBA()
        else:
            return self.YFBA()

    @cached(['designs', 'row_cov', 'col_cov', 'pheno'])
    def SrDWrFBALcWcCbar(self, i):
        if i < self.covar.Cr.getNumberParams():
            SgDWrFBALcWc = self.covar.Sg()[:, sp.newaxis] *  self.DWrFBALcWc()
        else:
            SgDWrFBALcWc = self.DWrFBALcWc()
        return sp.dot(SgDWrFBALcWc, self.covar.Cbar(i))

    @cached(['designs', 'row_cov', 'col_cov', 'pheno'])
    def WrRFBALcCtildeWc(self, i):
        if i < self.covar.Cr.getNumberParams():
            WrRFBALc = self.WrGGFBALc()
        else:
            WrRFBALc = self.WrFBALc()
        return sp.dot(WrRFBALc, sp.dot(self.covar.Ctilde(i), self.covar.Wc().T))

    # fixed effect methods for Areml_grad

    @cached('col_cov')
    def CtildeLcA(self, i):
        return sp.dot(self.covar.Ctilde(i), self.ALc().T)

    @cached(['designs', 'row_cov', 'col_cov'])
    def ALcCtildeLcA_o_FRF(self, i):
        if i < self.covar.Cr.getNumberParams():
            FRF = self.FGGF()
        else:
            FRF = self.FF()
        ALcCtildeALc = sp.dot(self.ALc(), self.CtildeLcA(i))
        return sp.kron(ALcCtildeALc, FRF)

    @cached(['designs', 'row_cov', 'col_cov'])
    def Cbar_o_Sr_dWLW(self, i):
        if i < self.covar.Cr.getNumberParams():
            Cbar_o_Sr = sp.kron(self.covar.Cbar(i), sp.diag(self.covar.Sg()))
        else:
            Cbar_o_Sr = sp.kron(self.covar.Cbar(i), sp.eye(self.covar.rank_r))
        return sp.dot(Cbar_o_Sr, self.dWLW())

    @cached(['designs', 'row_cov', 'col_cov'])
    def WcCtildeLcA_o_WrRF(self, i):
        if i < self.covar.Cr.getNumberParams():
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
        i = self.covar._actindex2index(i)
        R = sp.dot(self.WcCtildeLcA_o_WrRF(i).T, self.dWLW())
        R+= R.T
        R+= -self.ALcCtildeLcA_o_FRF(i)
        R+= -sp.dot(self.dWLW().T, self.Cbar_o_Sr_dWLW(i))
        return R

    ########################
    # LML terms
    ########################
    @cached(['designs', 'row_cov', 'col_cov', 'pheno'])
    def WKiy(self):
        R = vec(self.FYLcALc())
        # the following could be further optimized but do not care for now
        R -= sp.dot(self.WLW().T, vec(self.DWrYLcWc()))
        return R

    @cached(['row_cov', 'col_cov', 'designs', 'pheno'])
    def b(self):
        if self.mean.n_covs > 0:
            R = self.Areml.solve(self.WKiy())
        else:
            R = None
        return R

    @cached(['row_cov', 'col_cov', 'pheno'])
    def yKiy(self):
        return (self.YLc() * self.YLc()).sum() - (self.WrYLcWc() * self.DWrYLcWc()).sum()

    @cached(['designs', 'row_cov', 'col_cov', 'pheno'])
    def yKiWb(self):
        return (self.WKiy() * self.mean.b).sum()

    #########################
    # Gradients
    #########################
    @cached(['row_cov', 'col_cov', 'pheno'])
    def yKiy_grad_i(self,i):
        i = self.covar._actindex2index(i)
        r = - (self.YRY(i).T * self.covar.LcCtildeLc(i)).sum()
        r+= - (self.DWrYLcWc() * self.SrDWrYLcWcCbar(i)).sum()
        r+= 2 * (self.WrRYLcCtildeWc(i) * self.DWrYLcWc()).sum()
        return r

    @cached(['designs', 'row_cov', 'col_cov', 'pheno'])
    def yKiWb_grad_i(self,i):
        Areml_grad_i_b = sp.dot(self.Areml.K_grad_i(i), self.mean.b) 
        i = self.covar._actindex2index(i)
        #r = -2 * (self.YLc() * self.RFBALcCtilde(i)).sum()
        r = - 2 * (self.YRFBA(i).T * self.covar.LcCtildeLc(i)).sum()
        r+= -2 * (self.DWrYLcWc() * self.SrDWrFBALcWcCbar(i)).sum()
        r+= 2 * (self.WrRYLcCtildeWc(i) * self.DWrFBALcWc()).sum()
        r+= 2 * (self.WrRFBALcCtildeWc(i) * self.DWrYLcWc()).sum()
        r-= (self.mean.b * Areml_grad_i_b).sum()
        #r+= (self.FBALc() * self.RFBALcCtilde(i)).sum()
        #r+= (self.DWrFBALcWc() * self.SrDWrFBALcWcCbar(i)).sum()
        #r+= - 2 * (self.WrRFBALcCtildeWc(i) * self.DWrFBALcWc()).sum()
        return r
