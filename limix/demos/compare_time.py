import scipy as sp
import scipy.linalg as la
from limix.core.covar import Cov3KronSumLR
from limix.core.covar import FreeFormCov
from limix.core.gp import GP3KronSumLR
from limix.core.gp import GP
from limix.utils.preprocess import covar_rescale
#import old version of mtSet
import sys
sys.path.append('/Users/casale/Documents/mksum/mksum/mtSet_rev')
from mtSet.pycore.gp.gp3kronSum import gp3kronSum as gp3ks0
from mtSet.pycore.mean import mean
import mtSet.pycore.covariance as covariance
import time
import copy
import pdb

if __name__=='__main__':

    # define region and bg terms
    N = 2000
    f = 10
    P = 3
    G = 1.*(sp.rand(N, f)<0.2)
    X = 1.*(sp.rand(N, f)<0.2)
    R = covar_rescale(sp.dot(X,X.T))
    R+= 1e-4 * sp.eye(N)
    S, U = la.eigh(R)

    # define col covariances
    Cg = FreeFormCov(P, jitter=0)
    Cn = FreeFormCov(P)
    Cg.setRandomParams()
    Cn.setRandomParams()

    # define pheno
    Y = sp.randn(N, P)

    if 0:
        # debug covarianec
        cov = Cov3KronSumLR(Cn = Cn, Cg = Cg, S_R = S, U_R = U, G = G, rank = 1)
        cov.setRandomParams()
        pdb.set_trace()
        cov.K()
        print ((cov.H_chol_debug()-cov.H_chol())**2).mean()<1e-9
        print ((cov.inv_debug()-cov.inv())**2).mean()<1e-9
        print (cov.logdet_debug()-cov.logdet())**2
        print (cov.logdet_grad_i_debug(0)-cov.logdet_grad_i(0))**2

    # define GPs
    gp = GP3KronSumLR(Y = Y, Cg = Cg, Cn = Cn, S_R = S, U_R = U, G = G, rank = 1)
    gp0 = gp3ks0(mean(Y), covariance.freeform(P), covariance.freeform(P), S_XX=S, U_XX=U, rank=1)
    gp0.set_Xr(G)

#    gp._reset_profiler()

    pdb.set_trace()
    gp.covar.Sr()

    t0 = time.time()
    gp.LML()
    t1 = time.time()
    gp0.LML()
    t2 = time.time()
    print t1-t0
    print t2-t1

    for i in range(100):
        print i

        Cn.setRandomParams()
        params0 = {'Cr': gp.covar.Cr.getParams(), 'Cg': Cg.getParams(), 'Cn': Cn.getParams()}
        gp0.setParams(params0)

        gp.LML()
        gp0.LML()
        gp.LML_grad()
        gp0.LMLgrad()

    # get time profile
    tp = gp.get_time_profile().copy()
    tp.update(gp.covar.get_time_profile())

    # get time_in profile
    tpin = gp.get_timein_profile().copy()
    tpin.update(gp.covar.get_timein_profile())

    # get counter profile
    cp = gp.get_counter_profile().copy()
    cp.update(gp.covar.get_counter_profile())

    print '\nrotation of G'
    print gp0.time['cache_Xrchanged']
    print tp['Wr'], '(', tpin['Wr'], ')'
    print tp['Wr']/tpin['Wr']

    print '\nThe whole thing'
    print sp.sum(tp.values()), '(', sp.sum(tpin.values()), ')'

    import ipdb
    ipdb.set_trace()

    if 0:
        gp.LML()
        import pylab as pl
        pl.ion()
        pl.figure(1, figsize=(20,10))
        #gp.covar._profile(show=True)
        gp._profile(show=True, rot=90)
        pl.figure(2, figsize=(20,10))
        gp.covar._profile(show=True, rot=90)
        # ipdb.set_trace()




    # change params
    # import ipdb
    print 'Change Params covar:'
    # ipdb.set_trace()
    # import ipdb; ipdb.set_trace()
    gp.covar.diff(gp.covar.setRandomParams)
    print 'Change Params gp:'
    # ipdb.set_trace()
    gp.diff(gp.covar.setRandomParams)
    print 'Change G covar:'
    # ipdb.set_trace()
    gp.covar.diff(gp.covar.setG, 1.*(sp.rand(N, f)<0.2))
    print 'Change G gp:'
    # ipdb.set_trace()
    gp.diff(gp.covar.setG, 1.*(sp.rand(N, f)<0.2))
    # ipdb.set_trace()

    gp0 = GP(covar = copy.deepcopy(gp.covar), mean = copy.deepcopy(gp.mean))

    t0 = time.time()
    print 'GP2KronSum.LML():', gp.LML()
    print 'Time elapsed:', time.time() - t0

    # compare with normal gp
    # assess compatibility with this GP
    t0 = time.time()
    print 'GP.LML():', gp0.LML()
    print 'Time elapsed:', time.time() - t0

    t0 = time.time()
    print 'GP2KronSum.LML_grad():', gp.LML_grad()
    print 'Time elapsed:', time.time() - t0

    t0 = time.time()
    print 'GP.LML_grad():', gp0.LML_grad()
    print 'Time elapsed:', time.time() - t0

    pdb.set_trace()
    gp.optimize()
