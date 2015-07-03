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
import mtSet.pycore.optimize.optimize_bfgs as OPT
import time
import copy
import pdb
import h5py
import os
from limix.utils.util_functions import smartDumpDictHdf5

def gen_data(N=100, P=4):
    f = 20
    G = 1.*(sp.rand(N, f)<0.2)
    X = 1.*(sp.rand(N, f)<0.2)
    R = covar_rescale(sp.dot(X,X.T))
    R+= 1e-4 * sp.eye(N)
    S, U = la.eigh(R)
    Y = sp.randn(N, P)
    return Y, S, U, G 

if __name__=='__main__':

    P = 4

    # define col covariances
    Cg = FreeFormCov(P, jitter=0)
    Cn = FreeFormCov(P)
    Cg.setRandomParams()
    Cn.setRandomParams()

    out_file = './times.hdf5'

    if not os.path.exists(out_file) or 'recalc' in sys.argv:
        Ns = sp.array([100,150,200,300,500,800,1200,1600,2000,3000,4000,5000])
        n_rips = 5 
        t = sp.zeros((Ns.shape[0], n_rips))
        t0 = sp.zeros((Ns.shape[0], n_rips))
        r = sp.zeros((Ns.shape[0], n_rips))
        for ni, n in enumerate(Ns): 
            for ri in range(n_rips):
                print '.. %d individuals - rip %d' % (n, ri)
                print '   .. generating data'
                Y, S, U, G = gen_data(N=n, P=P)

                # define GPs
                gp = GP3KronSumLR(Y = Y, Cg = Cg, Cn = Cn, S_R = S, U_R = U, G = G, rank = 1)
                gp0 = gp3ks0(mean(Y), covariance.freeform(P), covariance.freeform(P), S_XX=S, U_XX=U, rank=1)
                gp0.set_Xr(G)
                gp._reset_profiler()

                if 1:
                    gp.covar.setRandomParams()
                else:
                    n_params = gp.covar.Cr.getNumberParams()
                    n_params+= gp.covar.Cg.getNumberParams()
                    n_params+= gp.covar.Cn.getNumberParams()
                    params1 = {'covar': sp.randn(n_params)}
                    gp.setParams(params1)
                params = {}
                params['Cr'] = gp.covar.Cr.getParams().copy()
                params['Cg'] = gp.covar.Cg.getParams().copy()
                params['Cn'] = gp.covar.Cn.getParams().copy()
                gp0.setParams(params)

                print '   .. optimization' 
                _t0 = time.time()
                conv, info = gp.optimize()
                _t1 = time.time()
                conv,info = OPT.opt_hyper(gp0,gp0.getParams())
                _t2 = time.time()
                t[ni, ri] = _t1-_t0
                t0[ni, ri] = _t2-_t1
                r[ni, ri] = t[ni, ri] / t0[ni, ri]
        RV = {'t': t, 't0': t0, 'r': r, 'Ns': Ns}
        fout = h5py.File(out_file, 'w')
        smartDumpDictHdf5(RV, fout)
        fout.close()
    else:
        R = {}
        fin = h5py.File(out_file, 'r')
        for key in fin.keys():
            R[key] = fin[key][:]
        fin.close()

    pdb.set_trace()

    import pylab as PL
    PL.subplot(211)
    PL.title('MTSet')
    PL.plot(R['Ns'], R['t'].mean(1),'g')
    PL.plot(R['Ns'], R['t0'].mean(1),'r')
    PL.ylabel('time')
    PL.subplot(212)
    PL.plot(R['Ns'], R['r'].mean(1))
    PL.ylabel('Time ratio')
    PL.xlabel('Number of samples')
    PL.savefig('mtset.pdf')
    PL.show()

    pdb.set_trace()

    #print '\n%d-th evaluation' % i
    #print 'LML'
    #t0 = time.time()
    #LML0 = gp.LML()
    #t1 = time.time()
    #LML1 = gp0.LML()
    #t2 = time.time()
    #print LML0-LML1
    #print 'gp new:', t1-t0
    #print 'gp old:', t2-t1
    #print (t1-t0)/(t2-t1), 'times slower'

    #print 'LMLgrad'
    #t0 = time.time()
    #gp.LML_grad()
    #t1 = time.time()
    #gp0.LMLgrad()
    #t2 = time.time()
    #print 'gp new:', t1-t0
    #print 'gp old:', t2-t1
    #print (t1-t0)/(t2-t1), 'times slower'

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
