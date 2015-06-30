import scipy as sp
import scipy.linalg as la
from limix.core.covar import Cov3KronSumLR
from limix.core.covar import FreeFormCov
from limix.core.gp import GP2KronSumLR
from limix.core.gp import GP
from limix.utils.preprocess import covar_rescale
#import old version of mtSet
import sys
sys.path.append('/Users/casale/Documents/mksum/mksum/mtSet_rev')
from mtSet.pycore.gp.gp2kronSumLR import gp2kronSumLR as gp2ks0
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
    F = sp.rand(N, 2)
    Y = sp.randn(N, P)
    return Y, F, G 

if __name__=='__main__':

    P = 4

    # define col covariances
    Cg = FreeFormCov(P, jitter=0)
    Cn = FreeFormCov(P)
    Cg.setRandomParams()
    Cn.setRandomParams()

    out_file = './times_PC.hdf5'

    if not os.path.exists(out_file) or 'recalc' in sys.argv:
        Ns = sp.array([100,150,200,300,500,800,1200,1600,2000,3000,4000,5000,6000,8000,10000,12000,14000,16000,20000,24000,32000,40000])
        n_rips = 5 
        t = sp.zeros((Ns.shape[0], n_rips))
        t0 = sp.zeros((Ns.shape[0], n_rips))
        r = sp.zeros((Ns.shape[0], n_rips))
        for ni, n in enumerate(Ns): 
            for ri in range(n_rips):
                print '.. %d individuals - rip %d' % (n, ri)
                print '   .. generating data'
                Y, F, G = gen_data(N=n, P=P)

                # define GPs
                gp = GP2KronSumLR(Y=Y, F=F, A=sp.eye(P), Cn=Cn, G=G)
                gp0 = gp2ks0(Y, covariance.freeform(P), F=F, rank=1)
                gp0.set_Xr(G)

                if 1:
                    gp.covar.setRandomParams()
                else:
                    n_params = gp.covar.Cr.getNumberParams()
                    n_params+= gp.covar.Cn.getNumberParams()
                    params1 = {'covar': sp.randn(n_params)}
                    gp.setParams(params1)
                params = {}
                params['Cr'] = gp.covar.Cr.getParams().copy()
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
    PL.title('MTSet-PC')
    PL.plot(R['Ns'], R['t'].mean(1),'g',label='new')
    PL.plot(R['Ns'], R['t0'].mean(1),'r',label='old')
    PL.plot(R['Ns'], R['t'].mean(1)-R['t0'].mean(1),'b',label='diff')
    PL.ylabel('time')
    PL.legend()
    PL.subplot(212)
    PL.plot(R['Ns'], R['r'].mean(1))
    PL.ylabel('Time ratio')
    PL.xlabel('Number of samples')
    PL.savefig('mtSetPC.pdf')
    PL.show()
    pdb.set_trace()

