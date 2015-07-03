import scipy as sp
import scipy.linalg as la
from limix.core.covar import FreeFormCov
from limix.core.covar import LowRankCov
#import old version of mtSet
import sys
sys.path.append('/Users/casale/Documents/mksum/mksum/mtSet_rev')
import mtSet.pycore.covariance as covariance
import time
import copy
import pdb
import h5py
import os
from limix.utils.util_functions import smartDumpDictHdf5

if __name__=='__main__':

    P = 4

    pdb.set_trace()

    # define col covariances
    C = FreeFormCov(P, jitter=0)
    C0 = covariance.freeform(P)

    t1 = 0
    t0 = 0

    for ti in range(1000):
        C.setRandomParams()
        C0.setParams(C.getParams())

        for i in range(int(0.5*P*(P+1))):
            _t0 = time.time()
            C0.Kgrad_param(i)
            _t1 = time.time()
            C.K_grad_i(i)
            _t2 = time.time()
            t0 += _t1-_t0
            t1 += _t2-_t1

    print 'old:', t0
    print 'new:', t1

    


