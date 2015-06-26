import scipy as sp
from limix.core.covar import Cov3KronSumLR
from limix.core.covar import FreeFormCov 
from limix.core.gp import GP3KronSumLR
from limix.core.gp import GP
from limix.utils.preprocess import covar_rescale
import time
import copy
import pdb

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

    pdb.set_trace()

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
    #gp.diff(gp.covar.setRandomParams)
    #pdb.set_trace()

    # change params
    #gp.covar.diff(gp.covar.setRandomParams)
    #gp.covar.Lr()
    #gp.covar.diff(gp.covar.setRandomParams)
    import ipdb
    ipdb.set_trace()
    gp.covar.diff(gp.covar.setG, 1.*(sp.rand(N, f)<0.2))
    ipdb.set_trace()
    gp.covar.Wr()
    gp.covar.diff(gp.covar.setG, 1.*(sp.rand(N, f)<0.2))
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

