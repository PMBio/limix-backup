import scipy as sp
from limix.core.covar import FreeFormCov
from limix.core.mean import MeanKronSum
from limix.core.gp import GP2KronSum
from limix.core.gp import GP
from limix.utils.preprocess import covar_rescale
import time
import copy
import pdb

if __name__=='__main__':

    # define phenotype
    N = 100
    P = 3
    Y = sp.randn(N,P)

    # define fixed effects
    F = []; A = []
    F.append(1.*(sp.rand(N,2)<0.5))
    #F.append(1.*(sp.rand(N,1)<0.5))
    A.append(sp.eye(P))
    #A.append(sp.ones((1,P)))

    # define row caoriance
    f = 10
    X = 1.*(sp.rand(N, f)<0.2)
    R = covar_rescale(sp.dot(X,X.T))
    R+= 1e-4 * sp.eye(N)

    # define col covariances
    Cg = FreeFormCov(P)
    Cn = FreeFormCov(P)
    Cg.setRandomParams()
    Cn.setRandomParams()

    # define gp
    pdb.set_trace()
    gp = GP2KronSum(Y = Y, F = F, A = A, Cg = Cg, Cn = Cn, XX = R)
    t0 = time.time()
    print 'GP2KronSum.LML():', gp.LML()
    print 'Time elapsed:', time.time() - t0

    # compare with normal gp
    # assess compatibility with this GP
    gp0 = GP(covar = copy.deepcopy(gp.covar), mean = copy.deepcopy(gp.mean))
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
    gp0.optimize()
    gp0.covar.setRandomParams()
    gp0.optimize()
    gp0.covar.setRandomParams()
    gp0.optimize()
    gp0.covar.setRandomParams()
    gp0.optimize()

    if 1:
        # check notification
        for i in range(10):
            gp.covar.setRandomParams()
            print gp.LML()
            print gp0.LML()
        

    if 0:
        # check LMLgrad terms
        print gp.Sr_DLrYLc_Ctilde(0)
        print gp.Sr_vei_dLWb_Ctilde(0)
        print gp.yKiy_grad_i(0)
        print gp.yKiWb_grad_i(0)
        print gp.Areml.K_grad_i(0)
        print gp0.Areml.K_grad_i(0)

    if 0:
        # check LML terms
        print gp.LrY()
        print gp.LrYLc()
        print gp.DLrYLc()
        print gp.LrF()
        print gp.ALc()
        print gp.LW()
        print gp.dLW()
        print gp.WKiy()
        gp.update_b()
        print gp.LML()

