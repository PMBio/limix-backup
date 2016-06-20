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

    ns = sp.array([100, 200, 400, 800, 1600, 2400, 3200, 4800])
    n_rips = 5

    time_ls = sp.zeros((ns.shape[0], n_rips))
    time_chol = sp.zeros((ns.shape[0], n_rips)) 
    err = sp.zeros((ns.shape[0], n_rips)) 
    for ni, n in enumerate(ns):
        for ri in range(n_rips):

            print('n = ', n, '- rip = ', ri)

            print('   .. generate data')
            y = sp.randn(n, 1)
            C = gen_cov(n)

            t0 = time.time()
            print('   .. solve chol')
            x_chol = C.solve(y)
            t1 = time.time()
            print('   .. solve linsys')
            x_ls   = C.solve_ls(y)
            t2 = time.time()

            time_chol[ni, ri] = t1 - t0
            time_ls[ni, ri] = t2 - t1
            err[ni, ri] = ((x_chol - x_ls)**2).mean()

    # check solution
    ipdb.set_trace()
    plt = pl.subplot(211)
    pl.plot(sp.log10(ns), sp.log10(time_chol.mean(1)), 'k')
    pl.plot(sp.log10(ns), sp.log10(time_ls.mean(1)), 'g')
    plt = pl.subplot(212)
    pl.plot(ns, err.mean(1), 'k')
    pl.show()
    

    

    
