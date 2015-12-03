import sys
sys.path.insert(0, '../..')
import scipy as sp
import scipy.linalg as la
from limix.core.covar import Cov3KronSumLR
from limix.core.covar import FreeFormCov 
from limix.core.covar import FixedCov 
from limix.core.gp import GP3KronSumLR
from limix.core.gp import GP
from limix.utils.preprocess import covar_rescale
import time
import copy
import pdb
import ipdb

sp.random.seed(2)

if __name__=='__main__':

    # define region and bg terms 
    N = 500
    f = 10
    P = 3
    G = 1.*(sp.rand(N, f)<0.2)
    X = 1.*(sp.rand(N, f)<0.2)
    R = covar_rescale(sp.dot(X,X.T))
    R+= 1e-4 * sp.eye(N)

    # define col covariances
    Cg = FreeFormCov(P)
    Cn = FreeFormCov(P)
    Cg.setRandomParams()
    Cn.setRandomParams()

    # define pheno
    Y = sp.randn(N, P)

    if 0:
        # debug covarianec
        cov = Cov3KronSumLR(Cn = Cn, Cg = Cg, R = R, G = G, rank = 1)
        cov.setRandomParams()
        pdb.set_trace()
        cov.K()
        print ((cov.H_chol_debug()-cov.H_chol())**2).mean()<1e-9
        print ((cov.inv_debug()-cov.inv())**2).mean()<1e-9
        print (cov.logdet_debug()-cov.logdet())**2
        print (cov.logdet_grad_i_debug(0)-cov.logdet_grad_i(0))**2

    # define GP
    gp = GP3KronSumLR(Y = Y, Cg = Cg, Cn = Cn, R = R, G = G, rank = 1)
    gp.optimize()
    ipdb.set_trace()

    # try optimzation with fixed covariance
    Cr = FixedCov(sp.ones((P,P)))
    gp = GP3KronSumLR(Y = Y, Cr=Cr, Cg = Cg, Cn = Cn, R = R, G = G, rank = 1)
    gp.optimize()
    ipdb.set_trace()

    if 0:
        # test fisher matrix
        n_seeds = 200
        for i in range(10):
            sp.random.seed(2)
            gp.covar.setRandomParams()
            Iexact = gp.covar._getIscoreTest(debug=True)
            print 'exact'
            print Iexact
            I1 = gp.covar._getIscoreTest(n_seeds=n_seeds, seed=i, debug1=True)
            print 'sample %d' % i
            print I1
            I2 = gp.covar._getIscoreTest(n_seeds=n_seeds, seed=i)
            print 'sample %d' % i
            print I2
        pdb.set_trace()

    if 1:
        # test score 
        n_seeds = 1000
        for i in range(10):
            gp.covar.setRandomParams()
            print gp.score(debug=True)
            print gp.score(n_seeds=n_seeds, seed=i)
        ipdb.set_trace()





    #gp.diff(gp.covar.setRandomParams)
    #pdb.set_trace()

if 0:

    import ipdb
    ipdb.set_trace()
    gp.LML()
    import pylab as pl
    pl.ion()
    pl.figure(1, figsize=(20,10))
    #gp.covar._profile(show=True)
    gp._profile(show=True, rot=90)
    pl.figure(2, figsize=(20,10))
    gp.covar._profile(show=True, rot=90)
    ipdb.set_trace()


    # change params
    import ipdb
    print 'Change Params covar:'
    ipdb.set_trace()
    gp.covar.diff(gp.covar.setRandomParams)
    print 'Change Params gp:'
    ipdb.set_trace()
    gp.diff(gp.covar.setRandomParams)
    print 'Change G covar:'
    ipdb.set_trace()
    gp.covar.diff(gp.covar.setG, 1.*(sp.rand(N, f)<0.2))
    print 'Change G gp:'
    ipdb.set_trace()
    gp.diff(gp.covar.setG, 1.*(sp.rand(N, f)<0.2))
    ipdb.set_trace()

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

