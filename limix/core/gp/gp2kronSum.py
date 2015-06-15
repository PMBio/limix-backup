import sys
from limix.core.mean import MeanKronSum
from limix.core.covar import Cov2KronSum
from limix.core.type.cached import *

import pdb
import numpy as NP
import scipy as sp
import scipy.linalg as LA
import time as TIME
from gp_base import GP
from limix.core.covar.cov_reml import cov_reml

class GP2KronSum(GP):

    def __init__(self,Y = None, F = None, A = None, Cg = None,Cn = None, XX=None, S_XX=None, U_XX=None):
        """
        Gaussian Process with a 2kronSum Covariance and a mean with kronecker terms(with REML)
        vec(Y) ~ N( vec( \sum_i A_i \kron F_i), Cg \kron R + Cn \kron I )
        ---------------------------------------------------------------------------
        Y:      Phenotype matrix
        Cg:     LIMIX trait-to-trait covariance for genetic contribution
        Cn:     LIMIX trait-to-trait covariance for noise
        XX:     Matrix for fixed sample-to-sample covariance function
        """
        print 'pass XX and S_XX to covariance: the covariance should be responsable of caching stuff'
        GP.__init__(self)
        self.covar = Cov2KronSum(Cg = Cg, Cn = Cn, R = XX)
        self.mean  = MeanKronSum(Y = Y, F = F, A = A)
        self.Areml = cov_reml(self)
        self.update_b()
        self.covar.register(self.col_cov_has_changed, 'row_cov')
        self.covar.register(self.col_cov_has_changed, 'col_cov')
        self.mean.register(self.pheno_has_changed, 'pheno')
        self.mean.register(self.designs_have_changed, 'designs')

    def col_cov_has_changed(self):
        self.clear_cache('LrYLc', 'DLrYLc', 'ALc', 'LW', 'dLW',
                            'WKiy', 'vei_dLWb', 'yKiy', 'yKiWb', 'Sr_DLrYLc_Ctilde',
                            'Sr_vei_dLWb_Ctilde', 'yKiy_grad_i', 'yKiWb_grad_i')
        self.clear_all()

    def row_cov_has_changed(self):
        self.clear_cache('LrY', 'LrYLc', 'DLrYLc', 'LrF', 'LW', 'dLW',
                            'WKiy', 'vei_dLWb', 'yKiy', 'yKiWb', 'Sr_DLrYLc_Ctilde',
                            'Sr_vei_dLWb_Ctilde', 'yKiy_grad_i', 'yKiWb_grad_i')
        self.clear_all()

    def pheno_has_changed(self):
        self.clear_cache('LrY', 'LrYLc', 'DLrYLc',
                            'WKiy', 'vei_dLWb', 'yKiy', 'yKiWb', 'Sr_DLrYLc_Ctilde',
                            'Sr_vei_dLWb_Ctilde', 'yKiy_grad_i', 'yKiWb_grad_i')
        self.clear_all()

    def designs_have_changed(self):
        self.clear_cache('LrF', 'ALc', 'LW', 'dLW', 'WKiy', 'vei_dLWb',
                            'yKiWb', 'Sr_vei_dLWb_Ctilde', 'yKiWb_grad_i')
        self.clear_all()


    ######################
    # Transformed phenotype
    ######################
    @cached
    def LrY(self):
        return sp.dot(self.covar.Lr(), self.mean.Y)

    @cached
    def LrYLc(self):
        return sp.dot(self.LrY(), self.covar.Lc().T)

    @cached
    def DLrYLc(self):
        return self.covar.D() * self.LrYLc()

    ######################
    # Transformed fixed effects
    ######################
    @cached
    def LrF(self):
        R = []
        for ti in range(self.mean.n_terms):
            R.append(sp.dot(self.covar.Lr(), self.mean.F[ti]))
        return R

    @cached
    def ALc(self):
        R = []
        for ti in range(self.mean.n_terms):
            R.append(sp.dot(self.mean.A[ti], self.covar.Lc().T))
        return R

    @cached
    def LW(self):
        R = sp.zeros((self.mean.Y.size, self.mean.n_covs))
        istart = 0
        for ti in range(self.mean.n_terms):
            iend = istart + self.mean.F[ti].shape[1] * self.mean.A[ti].shape[0]
            R[:, istart:iend] = sp.kron(self.ALc()[ti].T, self.LrF()[ti])
        return R

    @cached
    def dLW(self):
        return self.covar.d()[:,sp.newaxis] * self.LW()

    ######################
    # Areml
    ######################
    def Areml_K(self):
        return sp.dot(self.LW().T, self.dLW())

    def Areml_K_grad_i(self,i):
        dLWt = self.dLW().reshape((self.mean._N, self.mean._P, self.mean.n_covs), order = 'F')
        if i < self.covar.Cg.getNumberParams():
            SrdLWt = self.covar.Sr()[:, sp.newaxis, sp.newaxis] * dLWt
        else:
            SrdLWt = dLWt
        SrdLWtC = sp.tensordot(SrdLWt, self.covar.Ctilde(i), axes=(1, 1))
        SroCdLW = SrdLWtC.swapaxes(1,2).reshape((self.mean._N * self.mean._P, self.mean.n_covs), order = 'F')
        return -sp.dot(self.dLW().T, SroCdLW)

    ########################
    # LML terms
    ########################
    @cached
    def WKiy(self):
        R = sp.zeros((self.mean.n_covs, 1))
        istart = 0
        for ti in range(self.mean.n_terms):
            _dim = self.mean.F[ti].shape[1] * self.mean.A[ti].shape[0]
            iend = istart + _dim
            FLrDLrYLc = sp.dot(self.LrF()[ti].T, self.DLrYLc())
            R[istart:iend, 0] = sp.dot(FLrDLrYLc, self.ALc()[ti].T).reshape(_dim, order = 'F')
        return R

    def update_b(self):
        self.mean.b = self.Areml.solve(self.WKiy())

    @cached
    def vei_dLWb(self):
        # could be optimized but probably not worth it
        # as it requires a for loop
        r = sp.dot(self.dLW(), self.mean.b)
        return r.reshape(self.mean.Y.shape, order = 'F')

    @cached
    def yKiy(self):
        return (self.LrYLc()*self.DLrYLc()).sum()

    @cached
    def yKiWb(self):
        return (self.LrYLc() * self.vei_dLWb()).sum()

    #########################
    # Gradients
    #########################
    @cached
    def Sr_DLrYLc_Ctilde(self, i):
        return self.covar.Sr_X_Ctilde(self.DLrYLc(), i)

    @cached
    def Sr_vei_dLWb_Ctilde(self, i):
        return self.covar.Sr_X_Ctilde(self.vei_dLWb(), i)

    @cached
    def yKiy_grad_i(self,i):
        return -(self.DLrYLc() * self.Sr_DLrYLc_Ctilde(i)).sum()

    @cached
    def yKiWb_grad_i(self,i):
        rv = -2*(self.DLrYLc()*self.Sr_vei_dLWb_Ctilde(i)).sum()
        rv+= (self.vei_dLWb()*self.Sr_vei_dLWb_Ctilde(i)).sum()
        return rv
