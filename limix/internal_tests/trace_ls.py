import sys
#sys.path.insert(0, '/Users/casale/Documents/limix/limix/limix') 
sys.path.insert(0, '../..') 
import scipy as sp
from limix.core.covar import FixedCov
from limix.utils.preprocess import covar_rescale
import ipdb
import pylab as pl
import time

def gen_cov(n, r = 20):
    X = sp.randn(n, r)
    cov1 = covar_rescale(sp.dot(X, X.T))
    cov2 = covar_rescale(sp.diag(sp.rand(n)))
    cov  = covar_rescale(0.5 * cov1 + 0.5 * cov2)
    cov += 1e-4 * sp.eye(n)
    return FixedCov(cov)

if __name__=='__main__':

    sp.random.seed(0)

    ns = sp.array([500, 1000, 1500, 2000])
    nIterMCs = sp.array([2, 5, 10, 15, 20, 30])
    n_rips = 20

    err = sp.zeros((ns.shape[0], nIterMCs.shape[0], n_rips))
    t = sp.zeros((ns.shape[0], nIterMCs.shape[0], n_rips))
    for ni, n in enumerate(ns):
        for ii, nIterMC in enumerate(nIterMCs):
            for rip in range(n_rips):
                print(n, nIterMC, rip)
                C = gen_cov(n)
                C._nIterMC = nIterMC
                t0 = time.time()
                trace = C.sample_logdet_grad_i(0)
                err[ni, ii, rip] = 100. * abs(trace - n) / n
                t[ni, ii, rip] = time.time() - t0

    # check solution
    ipdb.set_trace()
    _n = err.shape[1]
    dloc = 1./(_n + 1.)
    width = 0.8 * dloc
    plt = pl.subplot(211)
    for i in range(_n):
        pl.boxplot(err[:,i,:].T, positions=sp.arange(4) + i * dloc, widths=width)
    plt.set_xticklabels(ns)
    pl.ylabel('perc error')
    pl.xlabel('sample size')
    plt = pl.subplot(212)
    for i in range(_n):
        pl.boxplot(t[:,i,:].T, positions=sp.arange(4) + i * dloc, widths=width)
    plt.set_xticklabels(ns)
    pl.ylabel('time (s)')
    pl.xlabel('sample size')
    pl.tight_layout()
    pl.savefig('sample_trace.pdf')
    pl.show()
    

    

    
