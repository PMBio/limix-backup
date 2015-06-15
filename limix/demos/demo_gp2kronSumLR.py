import scipy as sp
from limix.core.covar import Cov2KronSumLR
from limix.core.covar import FreeFormCov 
from limix.core.gp import GP2KronSumLR
from limix.core.gp import GP
from limix.utils.preprocess import covar_rescale
import time
import copy
import pdb

if __name__=='__main__':

    # define row caoriance
    N = 200
    f = 10
    P = 3
    X = 1.*(sp.rand(N, f)<0.2)

    # define col covariances
    Cn = FreeFormCov(P)

    if 0:
        # debug covarianec
        cov = Cov2KronSumLR(Cn = Cn, G = X, rank = 1)
        cov.setRandomParams()
        pdb.set_trace()
        print ((cov.inv_debug()-cov.inv())**2).mean()<1e-9
        print (cov.logdet_debug()-cov.logdet())**2
        print (cov.logdet_grad_i_debug(0)-cov.logdet_grad_i(0))**2

    # define fixed effects and pheno
    F = []; A = []
    F.append(1.*(sp.rand(N,2)<0.5))
    A.append(sp.eye(P))
    Y = sp.randn(N, P)

    gp = GP2KronSumLR(Y = Y, F = F, A = A, Cn = Cn, G = X)
    t0 = time.time()
    print 'GP2KronSum.LML():', gp.LML()
    print 'Time elapsed:', time.time() - t0

    # compare with normal gp
    # assess compatibility with this GP
    gp0 = GP(covar = copy.deepcopy(gp.covar), mean = copy.deepcopy(gp.mean))
    t0 = time.time()
    print 'GP.LML():', gp0.LML()
    print 'Time elapsed:', time.time() - t0

    print gp.LML_grad()['covar'] - gp0.LML_grad()['covar']

    print gp.yKiy_grad()
    print gp0.yKiy_grad()
    print gp.yKiWb_grad()
    print gp0.yKiWb_grad()
    print gp.Areml.K_grad_i(0)
    print gp0.Areml.K_grad_i(0)

    if 0:
        # test each term
        print gp.YLc()
        print gp.WrYLcWc()
        print gp.FY()
        print gp.FYLc()
        print gp.FYLcALc()
        print gp.FF()
        print gp.DWrYLcWc()
        print gp.WrF()
        print gp.ALc()
        print gp.ALcLcA()
        print gp.ALcWc()
        print gp.WLW()
        print gp.dWLW()
        print gp.Areml.K()
        print gp.WKiy()
        print gp.mean.b
        print gp.LML()
