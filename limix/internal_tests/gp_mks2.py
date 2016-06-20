import sys
sys.path.insert(0,'./../..')

from limix.core.covar import CovMultiKronSum
from limix.core.covar import FreeFormCov 
from limix.core.covar import SumCov 
from limix.core.covar import KronCov 
from limix.core.gp import GP
from limix.core.gp import GPLS 
from limix.core.gp import GPMKS 
from limix.core.mean import MeanBase
from limix.utils.preprocess import covar_rescale
from limix.utils.util_functions import vec

import ipdb
import scipy as sp
import scipy.linalg as LA
import time as TIME
import copy
import pylab as pl

sp.random.seed(2)

def generate_data(N, P, f, n_terms):
    Y = sp.zeros((N, P))
    R = []; C = []
    for term_i in range(n_terms):
        if term_i==n_terms-1:
            X = sp.eye(N)
        else:
            X = sp.zeros((N, N))
            X[:, :f] = 1.*(sp.rand(N, f)<0.2)
        W = sp.randn(P, P)
        Z = sp.randn(N, P)
        Z = sp.dot(X, sp.dot(Z, W.T))
        Z*= sp.sqrt(1. / (n_terms * Z.var(0).mean()))
        Y+= Z 
        _R = covar_rescale(sp.dot(X,X.T))
        if term_i!=n_terms-1:
            _R+= 1e-4 * sp.eye(N)
        R.append(_R)
        C.append(FreeFormCov(P))
    Y -= Y.mean(0)
    Y /= Y.std(0)
    return Y, C, R

def initCovars(C, value):
    for ti in range(len(C)):
        C[ti].setCovariance(value)
        

def check_equal(gpls_f, gpmks_f):
    y1 = gpls_f()
    y2 = gpmks_f()
    shape = list(y2.shape)
    if len(shape)>1:
        shape[0] = shape[0] * shape[1] 
        del shape[1]
    print(((y1-y2.reshape(shape, order='F'))**2).sum())

if __name__ == "__main__":

    # generate data
    N = 1000
    P = 2 
    f = 10
    n_terms = 3

    Y, C, R = generate_data(N, P, f, n_terms)
    y = Y.reshape((Y.size,1), order='F')
    C0v = sp.cov(Y.T) / n_terms
    initCovars(C, C0v)

    # standard sum of Kroneckers
    covar = SumCov(*[KronCov(C[i], R[i]) for i in range(len(C))])

    # define gps
    gp = GP(covar=covar, mean=MeanBase(y))
    gpls = GPLS(vec(Y), covar)
    gpmks = GPMKS(Y, C, R)

    # set nIterMC and tol for gpls and gpls and coordinate Zs
    n_seeds = 200 
    covar._nIterMC = n_seeds
    covar._tol = 1e-6
    gpmks.covar._nIterMC = n_seeds
    gpmks.covar._tol = 1e-6
    gpls.covar.Z()
    gpmks.covar.Z()
    gpmks.covar._cache_Z[:] = gpls.covar.Z().reshape((N, P, n_seeds), order='F')
    #gpls.covar._reuse = False
    ipdb.set_trace()
    

    if 1:
        # compares gps 
        n_times = 1
        for i in range(n_times):
            covar.setRandomParams()
            t0 = TIME.time()
            print(gp.LML_grad()['covar'])
            t1 = TIME.time()
            print(gpls.LML_grad()['covar'])
            t2 = TIME.time()
            print(gpmks.LML_grad()['covar'])
            t3 = TIME.time()
            print('gp_base:', t1-t0)
            print('gp_ls:', t2-t1)
            print('gp_mks:', t3-t2)
            ipdb.set_trace()
            #gpls.covar.resample(); print gpls.LML_grad()['covar']

    if 1:
        # optimize gp base
        initCovars(C, C0v)
        gp.optimize()
        print(C[0].K()) 
        print(C[1].K()) 
        print(C[2].K())
        ipdb.set_trace()

    if 1:
        # optimize linsys
        initCovars(C, C0v)
        gpls.optimize(debug=True,tr=0.1)
        print(C[0].K()) 
        print(C[1].K()) 
        print(C[2].K()) 
        ipdb.set_trace()

    if 1:
        # optimize linsys mks
        initCovars(C, C0v)
        gpmks.optimize(debug=True)
        print(C[0].K())
        print(C[1].K())
        print(C[2].K())
        ipdb.set_trace()

