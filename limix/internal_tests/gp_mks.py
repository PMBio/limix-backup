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
    N = 400
    P = 2 
    f = 10
    n_terms = 3
    n_seeds = 200

    Y, C, R = generate_data(N, P, f, n_terms)
    y = Y.reshape((Y.size,1))

    # standard sum of Kroneckers
    covar0 = SumCov(*[KronCov(C[i], R[i]) for i in range(len(C))])
    covar0.setRandomParams()
    covar0._nIterMC = n_seeds

    ipdb.set_trace()

    # specialized sum of Kroneckers
    ls = ['norot', 'rot', 'rot2']
    dot_method = ['std', 'kron']
    covard = {}
    for _ls in ls:
        for _dm in dot_method:
            key = _ls+'-'+_dm
            covard[key] = CovMultiKronSum(C, R, ls=_ls, dot_method=_dm) 
            covard[key]._nIterMC = n_seeds
    covar = covard['rot2-kron']

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
        z = sp.zeros(N*P*n_seeds)
        z[:] = Zt.transpose((2,1,0)).reshape(-1)
        t0 = TIME.time()
        ine = covar0.dot(Z)
        t1 = TIME.time()
        ce = covar.dot(Zt).reshape((N*P,n_seeds), order='F')
        t2 = TIME.time()
        print ((ine-ce)**2).sum()
        print 'time dot inefficient:', t1-t0
        print 'time dot efficient:', t2-t1
        print 'improvement:', (t1-t0) / (t2-t1)
        ipdb.set_trace()

    if 1:
        print 'Solve lin sys'
        Z = sp.randn(N*P, n_seeds)
        Zt = sp.zeros((N,P,n_seeds))
        Zt[:] = Z.reshape((N,P,n_seeds),order='F')
        t0 = TIME.time()
        KiZ_0 = covar0.solve_ls(Z)
        dt = TIME.time()-t0
        t = {}
        for key in covard.keys():
            t0 = TIME.time()
            KiZ = covard[key].solve_ls(Zt).reshape((N*P, n_seeds), order='F')
            t[key] = TIME.time() - t0
            print  ((KiZ-KiZ_0)**2).mean()
        print 'time dot inefficient:', dt
        for _ls in ls:
            for _dm in dot_method:
                key = _ls+'-'+_dm
                print key+':', t[key] 
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
        ipdb.set_trace()

    if 1:
        # define gps
        gpls = GPLS(vec(Y), covar0)
        gpmks = GPMKS(Y, C, R, nIterMC=n_seeds)

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
        
