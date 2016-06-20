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

    N = 500

    # define col covariances

    out_file = './times_scaleP.hdf5'

    if not os.path.exists(out_file) or 'recalc' in sys.argv:
        Ps = sp.array([2, 3, 4, 5])
        n_rips = 5 
        t = sp.zeros((Ps.shape[0], n_rips))
        t0 = sp.zeros((Ps.shape[0], n_rips))
        r = sp.zeros((Ps.shape[0], n_rips))
        for pi, p in enumerate(Ps): 
            for ri in range(n_rips):
                print('.. %d traits - rip %d' % (p, ri))
                print('   .. generating data')
                Y, S, U, G = gen_data(N=N, P=p)
                Cg = FreeFormCov(p, jitter=0)
                Cn = FreeFormCov(p)

                # define GPs
                gp = GP3KronSumLR(Y = Y, Cg = Cg, Cn = Cn, S_R = S, U_R = U, G = G, rank = 1)
                gp0 = gp3ks0(mean(Y), covariance.freeform(p), covariance.freeform(p), S_XX=S, U_XX=U, rank=1)
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

                print('   .. optimization') 
                _t0 = time.time()
                conv, info = gp.optimize()
                _t1 = time.time()
                conv,info = OPT.opt_hyper(gp0,gp0.getParams())
                _t2 = time.time()
                t[pi, ri] = _t1-_t0
                t0[pi, ri] = _t2-_t1
                r[pi, ri] = t[pi, ri] / t0[pi, ri]
        R = {'t': t, 't0': t0, 'r': r, 'Ps': Ps}
        fout = h5py.File(out_file, 'w')
        smartDumpDictHdf5(R, fout)
        fout.close()
    else:
        R = {}
        fin = h5py.File(out_file, 'r')
        for key in list(fin.keys()):
            R[key] = fin[key][:]
        fin.close()

    pdb.set_trace()

    import pylab as PL
    PL.title('MTSet')
    PL.subplot(211)
    PL.plot(R['Ps'], R['t'].mean(1),'g')
    PL.plot(R['Ps'], R['t0'].mean(1),'r')
    PL.ylabel('time')
    PL.subplot(212)
    PL.plot(R['Ps'], R['r'].mean(1))
    PL.ylabel('Time ratio')
    PL.xlabel('Number of traits')

    pdb.set_trace()

