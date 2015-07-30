import sys
sys.path.insert(0,'./../..')

from limix.core.covar import CovMultiKronSum
from limix.core.covar import FreeFormCov 
from limix.core.covar import SumCov 
from limix.core.covar import KronCov 
from limix.utils.preprocess import covar_rescale

import ipdb
import scipy as sp
import scipy.linalg as LA
import time as TIME
import copy
import pylab as pl

sp.random.seed(1)

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
        _R+= 1e-4 * sp.eye(N)
        R.append(_R)
        C.append(FreeFormCov(P))
    Y -= Y.mean(0)
    Y /= Y.std(0)
    return Y, C, R
        
if __name__ == "__main__":

    # generate data
    N = 1000
    P = 2 
    f = 10
    n_terms = 3

    Y, C, R = generate_data(N, P, f, n_terms)
    y = Y.reshape((Y.size,1))

    # standard sum of Kroneckers
    covar0 = SumCov(*[KronCov(C[i], R[i]) for i in range(len(C))])
    covar0.setRandomParams()

    # specialized sum of Kroneckers
    covar = CovMultiKronSum(C, R) 

    ipdb.set_trace()

    if 1:
        # basic checks
        print ((covar.K()-covar0.K())**2).mean()
        print ((covar.K_grad_i(0)-covar0.K_grad_i(0))**2).mean()
        print ((covar.K_hess_i_j(0, 0)-covar0.K_hess_i_j(0, 0))**2).mean()
        ipdb.set_trace()

    if 1:
        # check linear system solve
        Z = sp.randn(N*P, 30)
        Zt = sp.zeros((N,P,30))
        Zt[:] = Z.reshape((N,P,30),order='F')
        t0 = TIME.time()
        KiZ = covar.solve_ls(Z)
        t1 = TIME.time()
        KiZ_0 = covar0.solve_ls(Z)
        t2 = TIME.time()
        KiZ_1 = covar.solve_ls1(Zt).reshape((N*P, 30), order='F')
        t3 = TIME.time()
        print ((KiZ-KiZ_0)**2).mean()
        print ((KiZ-KiZ_1)**2).mean()
        print 'time dot efficient:', t1-t0
        print 'time dot inefficient:', t2-t1
        print 'time dot efficient new:', t3-t2
        print 'improvement:', (t2-t1) / (t1-t0)
        ipdb.set_trace()


    if 1:
        # coordinate zetas
        Z = covar.Z()
        covar0.Z()
        covar0._cache_Z[:] = covar.Z().reshape((N*P, 30), order='F')

        # checking DKZ function 
        t0 = TIME.time()
        DKZ0 = covar0.DKZ()
        t1 = TIME.time()
        DKZ = covar.DKZ().reshape(DKZ0.shape, order='F')
        t2 = TIME.time()
        print ((DKZ-DKZ0)**2).mean()
        print 'time dot inefficient:', t1-t0
        print 'time dot efficient:', t2-t1
        print 'improvement:', (t1-t0) / (t2-t1)
        ipdb.set_trace()

        # checking DDKZ function
        t0 = TIME.time()
        DDKZ0 = covar0.DDKZ()
        t1 = TIME.time()
        DDKZ = covar.DDKZ().reshape(DDKZ0.shape, order='F')
        t2 = TIME.time()
        print ((DDKZ-DDKZ0)**2).mean()
        print 'time dot inefficient:', t1-t0
        print 'time dot efficient:', t2-t1
        print 'improvement:', (t1-t0) / (t2-t1)
        ipdb.set_trace()
        
