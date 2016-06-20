import sys
sys.path.insert(0,'./../../limix')
import limix
from limix.core.mean.mean_base import MeanBase as lin_mean
from limix.core.covar import CategoricalLR
from limix.core.covar import FreeFormCov
from limix.core.covar import LowRankCov
from limix.core.covar import FixedCov
from limix.core.gp import GP
from limix.mtSet.core.iset_utils import *
from limix.utils.util_functions import smartDumpDictHdf5
import scipy as sp
import scipy.linalg as la
import pdb
import time
import os
import h5py


def generate_pheno(Cr, Cn, G, covs, W):
    Cr_h = msqrt(Cr)
    Cn_h = msqrt(Cn)
    Y = sp.dot(covs, W)
    Y+= sp.dot(G, sp.dot(sp.randn(G.shape[1], Cr_h.shape[1]), Cr_h.T))
    Y+= sp.dot(sp.randn(Y.shape[0], Cn_h.shape[1]), Cn_h.T)
    return Y

class ISet_Strat():

    def __init__(self, y, Ie, Xr, covs=None):
        """
        Args:
            y:          [N, 1] phenotype vector
            Ie:         N boolean context indicator
            U_R:        [N, N] eigenvector matrix of R
            Xr:         [N, S] genotype data of the set component
            covs:       [N, K] matrix for K covariates
        """
        if covs is None:
            covs = sp.ones((y.shape[0], 1))
        # kroneckerize covs
        W = sp.zeros((y.shape[0], 2*covs.shape[1]))
        W[:, :covs.shape[1]] = Ie[:, sp.newaxis] * covs 
        W[:, covs.shape[1]:] = (~Ie[:, sp.newaxis]) * covs 

        # store stuff 
        self.Ie  = Ie
        self.Xr  = Xr
        self.covs = covs
        self.mean = lin_mean(y, W) 

    def fitNull(self, out_dir='./cache', fname=None):

        if fname is not None:
            out_file = os.path.join(out_dir, fname)
            if os.path.exists(out_file):
                fo = h5py.File(out_file, 'r')
                null = {}
                for key in list(fo.keys()):
                    null[key] = fo[key][:]
                fo.close()
                self.null = null
                return null

        #1. build model
        Cr = FixedCov(sp.ones([2, 2]))
        Cr.scale = 1e-9
        Cr.act_scale = False
        covar = CategoricalLR(Cr, sp.ones((self.Xr.shape[0], 1)), self.Ie)
        gp = GP(covar=covar, mean=self.mean)

        #2. initialize
        covar.Cn.setCovariance(sp.eye(2))

        #3. optimize
        t0 = time.time()
        conv, _ = gp.optimize(verbose=False)
        t = time.time() - t0

        weights = gp.mean.b.reshape((self.covs.shape[1], 2), order='F')
        null = {'conv': sp.array([conv]),
                'time': sp.array([t]),
                'NLL': sp.array([gp.LML()]),
                'Cn': covar.Cn.K(),
                'weights': weights
                }

        if fname is not None:
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            fo = h5py.File(out_file, 'w')
            smartDumpDictHdf5(null, fo)
            fo.close()

        self.null = null
        return null

    def fit(self, region_covar_type='freeform', Cr_0=None):
        #1. define local covariance
        if region_covar_type=='block':
            Cr = FixedCov(sp.ones([2, 2]))
        elif region_covar_type=='lowrank':
            Cr = LowRankCov(2, rank=1)
        elif region_covar_type=='freeform':
            Cr = FreeFormCov(2, jitter=0.)

        #2. build model
        covar = CategoricalLR(Cr, self.Xr, self.Ie)
        covar.W_grad_i(0)
        gp = GP(covar=covar, mean=self.mean)

        if Cr_0 is None:
            if region_covar_type in ['block', 'lowrank']:
                Cr_0 = self.full['Cr']
            elif region_covar_type=='freeform':
                Cr_0 = 1e-4 * sp.ones((2, 2)) + 1e-4 * sp.eye(2)

        #3. initialize
        if region_covar_type=='block':
            Cr.scale = Cr_0.diagonal().mean() 
        elif region_covar_type=='lowrank':
            Cr.setCovariance(Cr_0)
        elif region_covar_type=='freeform':
            Cr.setCovariance(Cr_0)
        Cn_0 = self.null['Cn']
        covar.Cn.setCovariance(Cn_0)

        #4. optimize
        t0 = time.time()
        conv, _ = gp.optimize(verbose=False)
        t = time.time() - t0

        weights = gp.mean.b.reshape((self.covs.shape[1], 2), order='F')
        RV = {'conv': sp.array([conv]),
                'time': sp.array([t]),
                'NLL': sp.array([gp.LML()]),
                'Cr': Cr.K(),
                'Cn': covar.Cn.K(), 
                'weights': weights 
                }

        # var comps
        RV['var_r'] = sp.array([var_CoXX(RV['Cr'], self.Xr)])
        RV['var_n'] = sp.array([var_CoXX(RV['Cn'], sp.eye(self.Xr.shape[0]))])
        RV['var_c'] = sp.dot(self.mean.W, gp.mean.b).var(0)

        if region_covar_type=='freeform':
            self.full = RV
        if region_covar_type=='lowrank':
            self.lr = RV
        if region_covar_type=='block':
            self.block = RV

        return RV

    def fitBlock(self):
        return self.fit(region_covar_type='block')

    def fitFullRank(self):
        return self.fit(region_covar_type='freeform')

    def fitLowRank(self):
        return self.fit(region_covar_type='lowrank')

    def getVC(self):
        """
        Variance componenrs
        """
        _Cr = decompose_GxE(self.full['Cr'])
        RV = {}
        for key in list(_Cr.keys()):
            RV['var_%s' % key] = sp.array([var_CoXX(_Cr[key], self.Xr)])
        RV['var_c'] = self.full['var_c'] 
        RV['var_n'] = self.full['var_n'] 
        return RV

    def _sim_from(self, set_covar='rank1', qq=False, seed=None, ro_env=False):
        if set_covar=='rank1':      table = self.lr
        if set_covar=='block':      table = self.block
        if seed is not None:
            sp.random.seed(seed)
        Y = generate_pheno(table['Cr'], table['Cn'], self.Xr, self.covs, table['weights'])
        y = sp.zeros((Y.shape[0], 1))
        y[self.Ie, 0] = Y[self.Ie, 0]
        y[~self.Ie, 0] = Y[~self.Ie, 1]
        if ro_env:
            y = gaussianize(y)
            y = regressOut(y, (1.*self.Ie[:,sp.newaxis]))
        if ro_env or qq:
            y = gaussianize(y)
            y-= y.mean(0)
            y/= y.std(0)
        return y

