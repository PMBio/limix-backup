import sys
sys.path.insert(0,'./../..')

from limix.core.covar import CovMultiKronSum
from limix.core.covar import FreeFormCov 
from limix.core.covar import SumCov 
from limix.core.covar import KronCov 
from limix.core.gp import GPLS 
from limix.core.gp import GPMKS 
from limix.utils.preprocess import covar_rescale
from limix.utils.util_functions import vec

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
        if term_i!=n_terms-1:
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
    n_seeds = 30

    Y, C, R = generate_data(N, P, f, n_terms)
    y = Y.reshape((Y.size,1))

    # standard sum of Kroneckers
    covar0 = SumCov(*[KronCov(C[i], R[i]) for i in range(len(C))])
    covar0.setRandomParams()

    ipdb.set_trace()

    # specialized sum of Kroneckers
    covar = CovMultiKronSum(C, R) 
    covar_rot = CovMultiKronSum(C, R, ls='rot')
    covar_rot2 = CovMultiKronSum(C, R, ls='rot2')

    ipdb.set_trace()

    if 1:
        # basic checks
        print ((covar.K()-covar0.K())**2).mean()
        print ((covar.K_grad_i(0)-covar0.K_grad_i(0))**2).mean()
        print ((covar.K_hess_i_j(0, 0)-covar0.K_hess_i_j(0, 0))**2).mean()
        ipdb.set_trace()

    if 1:
        print 'One single dot'
        Z = sp.randn(N*P, n_seeds)
        Zt = sp.zeros((N,P,n_seeds))
        Zt[:] = Z.reshape((N,P,n_seeds),order='F')
        t0 = TIME.time()
        covar0.dot(Z)
        t1 = TIME.time()
        covar.dot_NxPxS(Zt)
        t2 = TIME.time()
        print 'time dot inefficient:', t1-t0
        print 'time dot efficient:', t2-t1
        print 'improvement:', (t1-t0) / (t2-t1)
        ipdb.set_trace()

    if 0:
        ipdb.set_trace()
        import scipy.linalg as LA
        Ki1 = LA.inv(covar.K())
        T = sp.kron(covar.C[-1].USi2(), sp.eye(N))
        Ki2 = sp.dot(T, sp.dot(LA.inv(covar.Kt()), T.T))
        print ((K1i-Ki2)**2).mean()
        Z = sp.randn(N*P, n_seeds)
        Zt = sp.zeros((N,P,n_seeds))
        Zt[:] = Z.reshape((N,P,n_seeds),order='F')
        KtZ = sp.dot(covar.Kt(), Z)
        KtZ1 = covar.dot_NxPxS_rot(Zt).reshape(KtZ.shape, order='F')
        print ((KtZ-KtZ1)**2).mean()
        KiZ = sp.dot(Ki1, Z)
        _Z = sp.tensordot(Zt, covar.C[-1].USi2(), (1,0)).transpose((0,2,1)).reshape(Z.shape, order='F')
        __Z = sp.dot(LA.inv(covar.Kt()), _Z).reshape(Zt.shape, order='F')
        KiZ2 = sp.tensordot(__Z, covar.C[-1].USi2(), (1,1)).transpose((0,2,1)).reshape(Z.shape, order='F')
        

    if 1:
        print 'Solve lin sys'
        Z = sp.randn(N*P, n_seeds)
        Zt = sp.zeros((N,P,n_seeds))
        Zt[:] = Z.reshape((N,P,n_seeds),order='F')
        t0 = TIME.time()
        KiZ_0 = covar0.solve_ls(Z)
        t1 = TIME.time()
        KiZ = covar.solve_ls_NxPxS(Zt).reshape((N*P, n_seeds), order='F')
        t2 = TIME.time()
        KiZr = covar_rot.solve_ls_NxPxS(Zt).reshape((N*P, n_seeds), order='F')
        t3 = TIME.time()
        KiZr2 = covar_rot2.solve_ls_NxPxS(Zt).reshape((N*P, n_seeds), order='F')
        t4 = TIME.time()
        print ((KiZ-KiZ_0)**2).mean()
        print ((KiZ-KiZr)**2).mean()
        print ((KiZ-KiZr2)**2).mean()
        print 'time dot inefficient:', t1-t0
        print 'time dot efficient (default):', t2-t1
        print 'time dot efficient (rot):', t3-t2
        print 'time dot efficient (rot2):', t4-t3
        #print 'improvement:', (t2-t1) / (t1-t0)
        ipdb.set_trace()

    # coordinate zetas
    Z = covar.Z()
    covar0.Z()
    covar0._cache_Z[:] = covar.Z().reshape((N*P, n_seeds), order='F')

    if 1:
        print 'DKZ'
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

        print 'DDKZ'
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

    if 1:
        # test logdet and trace functions
        logdet = covar.sample_logdet_grad()
        logdet0 = covar0.sample_logdet_grad()
        print ((logdet-logdet0)**2).mean()

        tr = covar.sample_trKiDDK()
        tr0 = covar0.sample_trKiDDK()
        print ((tr-tr0)**2).mean()

    if 1:
        # define gps
        gpls = GPLS(vec(Y), covar0)
        gpmks = GPMKS(Y, C, R)

        # sync Z
        gpmks.covar.Z()
        gpmks.covar._cache_Z[:] = covar.Z()

        print 'Kiy' 
        t0 = TIME.time()
        Kiy = gpmks.Kiy()
        t1 = TIME.time()
        Kiy0 = gpls.Kiy()
        t2 = TIME.time()
        print ((Kiy-Kiy0)**2).mean()
        print 'time dot efficient:', t1-t0
        print 'time dot inefficient:', t2-t1
        print 'improvement:', (t2-t1) / (t1-t0)
        ipdb.set_trace()


        print 'DKKiy' 
        t0 = TIME.time()
        DKKiy = gpmks.DKKiy()
        t1 = TIME.time()
        DKKiy0 = gpls.DKKiy()
        t2 = TIME.time()
        print ((DKKiy-DKKiy0)**2).mean()
        print 'time dot efficient:', t1-t0
        print 'time dot inefficient:', t2-t1
        print 'improvement:', (t2-t1) / (t1-t0)
        ipdb.set_trace()

        print 'DDKKiy' 
        t0 = TIME.time()
        DDKKiy = gpmks.DDKKiy()
        t1 = TIME.time()
        DDKKiy0 = gpls.DDKKiy()
        t2 = TIME.time()
        print ((DDKKiy-DDKKiy0)**2).mean()
        print 'time dot efficient:', t1-t0
        print 'time dot inefficient:', t2-t1
        print 'improvement:', (t2-t1) / (t1-t0)
        ipdb.set_trace()

        print 'AIM' 
        t0 = TIME.time()
        AIM = gpmks.AIM()
        t1 = TIME.time()
        AIM0 = gpls.AIM()
        t2 = TIME.time()
        print ((AIM-AIM0)**2).mean()
        print 'time dot efficient:', t1-t0
        print 'time dot inefficient:', t2-t1
        print 'improvement:', (t2-t1) / (t1-t0)
        ipdb.set_trace()

        print ((gpmks.LML()-gpls.LML())**2)
        print ((gpmks.LML_grad()['covar'] - gpls.LML_grad()['covar'])**2).mean()
        ipdb.set_trace()
        
