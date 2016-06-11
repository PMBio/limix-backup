import sys
sys.path.insert(0,'./../../limix')
import limix
from limix.core.mean.mean_base import MeanBase as lin_mean
from limix.core.covar import CategoricalLR
from limix.core.covar import FreeFormCov
from limix.core.covar import LowRankCov
from limix.core.covar import FixedCov
from limix.core.gp import GP
from include.vardecomp_gxe_utils import * 
import scipy as sp
import scipy.linalg as la
import pdb
import time
import os
import h5py
sys.path.append('./..')
from include.utils import smartDumpDictHdf5
from vardecomp_gxe_utils import decompose_GxE


def generate_pheno(Cr, Cn, G, covs, W):
    Cr_h = msqrt(Cr)
    Cn_h = msqrt(Cn)
    Y = sp.dot(covs, W)
    Y+= sp.dot(G, sp.dot(sp.randn(G.shape[1], Cr_h.shape[1]), Cr_h.T))
    Y+= sp.dot(sp.randn(Y.shape[0], Cn_h.shape[1]), Cn_h.T)
    return Y

class ISet_Strat():

    def __init__(self, y, Ie, G, covs=None):
        if covs is None:
            covs = sp.ones((y.shape[0], 1))
        # kroneckerize covs
        W = sp.zeros((y.shape[0], 2*covs.shape[1]))
        W[:, :covs.shape[1]] = Ie[:, sp.newaxis] * covs 
        W[:, covs.shape[1]:] = (~Ie[:, sp.newaxis]) * covs 

        # store stuff 
        self.Ie  = Ie
        self.G   = G
        self.covs = covs
        self.mean = lin_mean(y, W) 

    def fit_null(self, out_dir='./cache', fname=None):

        if fname is not None:
            out_file = os.path.join(out_dir, fname)
            if os.path.exists(out_file):
                fo = h5py.File(out_file, 'r')
                null = {}
                for key in fo.keys():
                    null[key] = fo[key][:]
                fo.close()
                self.null = null
                return null

        #1. build model
        Cr = FixedCov(sp.ones([2, 2]))
        Cr.scale = 1e-9
        Cr.act_scale = False
        covar = CategoricalLR(Cr, sp.ones((self.G.shape[0], 1)), self.Ie)
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
                'LML': sp.array([gp.LML()]),
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
        covar = CategoricalLR(Cr, self.G, self.Ie)
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
                'LML': sp.array([gp.LML()]),
                'Cr': Cr.K(),
                'Cn': covar.Cn.K(), 
                'weights': weights 
                }

        # var comps
        RV['var_r'] = sp.array([var_CoXX(RV['Cr'], self.G)])
        RV['var_n'] = sp.array([var_CoXX(RV['Cn'], sp.eye(self.G.shape[0]))])
        RV['var_c'] = sp.dot(self.mean.W, gp.mean.b).var(0)

        if region_covar_type=='freeform':
            self.full = RV
        if region_covar_type=='lowrank':
            self.lr = RV
        if region_covar_type=='block':
            self.block = RV

        return RV

    def getVCs(self):
        """
        This function was used in simulations and only
        now it has been updated 
        """
        if 0:
            RV = sp.zeros(5)
            #1. region contributions 
            Crs = decompose_GxE(self.full['Cr'])
            for ic, Cr in enumerate(Crs):
                L = msqrt(Cr)
                W = sp.zeros((self.G.shape[0], 2*self.G.shape[1]))
                W[self.Ie] = sp.kron(L[0], self.G[self.Ie])
                W[~self.Ie] = sp.kron(L[1], self.G[~self.Ie])
                PW = W-W.mean(0)
                RV[ic] = (PW**2).sum() / float(self.G.shape[0]-1)
            #1. contrubtion from covariates
            RV[3] = sp.dot(self.mean.W, self.mean.b).var(0)
            #2. noise 
            d = sp.zeros(self.G.shape[0])
            d[self.Ie] = self.full['Cn'][0,0]
            d[~self.Ie] = self.full['Cn'][1,1]
            RV[4] = d.mean()
            return RV
        else:
            return self.getVC()

    def getVC(self):
        """
        Variance componenrs
        """
        _Cr = decompose_GxE(self.full['Cr'])
        _Cr['r'] = self.full['Cr']
        RV = {}
        for key in _Cr.keys():
            RV['var_%s' % key] = sp.array([var_CoXX(_Cr[key], self.G)])
        RV['var_c'] = self.full['var_c'] 
        RV['var_n'] = self.full['var_n'] 
        return RV


    def _sim_from(self, gen_model='lr', qq=False, seed=None, ro_env=False):
        if gen_model=='lr':         table = self.lr
        if gen_model=='block':      table = self.block
        if seed is not None:
            sp.random.seed(seed)
        Y = generate_pheno(table['Cr'], table['Cn'], self.G, self.covs, table['weights'])
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

