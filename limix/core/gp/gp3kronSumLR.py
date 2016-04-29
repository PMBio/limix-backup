import sys
from limix.core.mean import MeanKronSum
from limix.core.covar import Cov3KronSumLR
from hcache import Cached, cached
from limix.utils.util_functions import vec

import pdb
import numpy as NP
import scipy as sp
import scipy.linalg as la
import time as TIME
from .gp_base import GP
from limix.core.gp import GP2KronSum
from limix.core.covar.cov_reml import cov_reml

class GP3KronSumLR(GP2KronSum):
    """
    Gaussian Process with a 3kronSumLR Covariance and a mean that is a sum of Kronecker products:
        vec(Y) ~ N( vec( \sum_i F_i B_i A_i), Cr \kron GG.T + Cg \kron R + Cn \kron I )
    The current implementation of this class does not support fixed effects!
    Notation:
        N = number of samples
        P = number of traits
        Y = [N, P] phenotype matrix
        F_i = sample fixed effect design for term i
        A_i = trait fixed effect design for term i
        B_i = effect sizes of fixed effect term i
        Cr = column covariance matrix for low-rank term 
        Cg = column covariance matrix for full rank non-noise term
        Cn = column covariance matrix for noise term
        R = row covariance matrix for non-noise term
        rank_c = rank of low-rank col covariance
        rank_r = rank of low-rank row covariance
    """

    def __init__(self,Y = None, Cg = None, Cn = None, R = None, S_R = None, U_R = None, G = None, rank = None):
        """
        Args:
            Y:      [N, P] phenotype matrix
            Cg:     Limix covariance matrix for Cg (dimension P)
            Cn:     Limix covariance matrix for Cn (dimension P)
            G:      [N, rank_r] numpy covariance matrix for G
            R:      [N, N] numpy semidemidefinite covariance matrix for R.
                    In alternative to R, S_R and U_R can be specified.
            S_R:    N vector of eigenvalues of R
            U_R:    [N, N] eigenvector matrix of R
            rank:   rank of column low-rank covariance (default = 1)
        """
        covar = Cov3KronSumLR(Cg=Cg, Cn=Cn, R=R, G=G, rank=rank, S_R=S_R, U_R=U_R)
        mean  = MeanKronSum(Y = Y)
        GP.__init__(self, covar = covar, mean = mean)

    def _observe(self):
        self.covar.register(self.row_cov_has_changed, 'row_cov')
        self.covar.register(self.col_cov_has_changed, 'col_cov')
        self.covar.register(self.G_has_changed, 'G')
        self.mean.register(self.pheno_has_changed, 'pheno')
        self.mean.register(self.designs_have_changed, 'designs')

    def G_has_changed(self):
        self.clear_cache('G')
        self.clear_all()

    ######################
    # Transformed phenotype
    ######################
    @cached(['pheno','col_cov', 'row_cov', 'G'])
    def ve_WrDLrYLcWc(self):
        return vec(sp.dot(self.covar.Wr().T, sp.dot(self.DLrYLc(), self.covar.Wc())))

    @cached(['pheno','col_cov', 'row_cov', 'G'])
    def Hi_ve_WrDLrYLcWc(self):
        return la.cho_solve((self.covar.H_chol(), True), self.ve_WrDLrYLcWc())

    @cached(['pheno','col_cov', 'row_cov', 'G'])
    def vei_HiveWrDLrYLcWc(self):
        return self.Hi_ve_WrDLrYLcWc().reshape((self.covar.rank_r, self.covar.rank_c), order = 'F')

    @cached(['pheno','col_cov','row_cov','G'])
    def DLrYLc_m_DWr_veiHiveWrDLrYLcWc_Wc(self):
        R  = sp.dot(self.covar.Wr(), sp.dot(self.vei_HiveWrDLrYLcWc(), self.covar.Wc().T))
        R *= -self.covar.D()
        R += self.DLrYLc()
        return R

    @cached(['pheno','col_cov','row_cov','G'])
    def WrWr_DLrYLc_m_DWr_veiHiveWrDLrYLcWc_Wc(self):
        return sp.dot(self.covar.Wr(), sp.dot(self.covar.Wr().T, self.DLrYLc_m_DWr_veiHiveWrDLrYLcWc_Wc()))

    @cached(['pheno','col_cov','row_cov','G'])
    def Sr_DLrYLc_m_DWr_veiHiveWrDLrYLcWc_Wc(self):
        return self.covar.Sr()[:, sp.newaxis] * self.DLrYLc_m_DWr_veiHiveWrDLrYLcWc_Wc()

    @cached(['pheno','col_cov','row_cov','G'])
    def Rtilde_DLrYLcmDWrveiHiveWrDLrYLcWcWc_Ctilde(self, i):
        np_r = self.covar.Cr.getNumberParams()
        np_g = self.covar.Cg.getNumberParams()
        if i < np_r:
            R = self.WrWr_DLrYLc_m_DWr_veiHiveWrDLrYLcWc_Wc() 
        elif i < (np_r + np_g):
            R = self.Sr_DLrYLc_m_DWr_veiHiveWrDLrYLcWc_Wc() 
        else:
            R = self.DLrYLc_m_DWr_veiHiveWrDLrYLcWc_Wc()
        return sp.dot(R, self.covar.Ctilde(i))

    @cached(['row_cov', 'col_cov', 'pheno'])
    def Sr_DLrYLc_Ctilde(self, i):
        pass

    @cached(['row_cov', 'col_cov', 'designs', 'pheno'])
    def Sr_vei_dLWb_Ctilde(self, i):
        pass

    ######################
    # Areml
    ######################
    def Areml_K(self):
        pass

    def Areml_K_grad_i(self,i):
        pass

    ########################
    # LML terms
    ########################
    @cached(['row_cov', 'col_cov', 'designs', 'pheno'])
    def WKiy(self):
        pass

    @cached(['row_cov', 'col_cov', 'pheno'])
    def yKiy(self):
        r = (self.LrYLc()*self.DLrYLc()).sum()
        r-= (self.ve_WrDLrYLcWc() * self.Hi_ve_WrDLrYLcWc()).sum()
        return r

    @cached(['row_cov', 'col_cov', 'designs', 'pheno'])
    def yKiWb(self):
        pass

    #########################
    # Gradients
    #########################
    @cached(['row_cov', 'col_cov', 'pheno'])
    def yKiy_grad_i(self,i):
        return -sp.sum(self.DLrYLc_m_DWr_veiHiveWrDLrYLcWc_Wc() * self.Rtilde_DLrYLcmDWrveiHiveWrDLrYLcWcWc_Ctilde(i))

    @cached(['row_cov', 'col_cov', 'designs', 'pheno'])
    def yKiWb_grad_i(self,i):
        pass
