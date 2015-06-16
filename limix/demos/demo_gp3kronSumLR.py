import scipy as sp
from limix.core.covar import Cov3KronSumLR
from limix.core.covar import FreeFormCov 
from limix.core.gp import GP2KronSumLR
from limix.core.gp import GP
from limix.utils.preprocess import covar_rescale
import time
import copy
import pdb

if __name__=='__main__':

    # define region and bg terms 
    N = 200
    f = 10
    P = 3
    G = 1.*(sp.rand(N, f)<0.2)
    X = 1.*(sp.rand(N, f)<0.2)
    R = covar_rescale(sp.dot(X,X.T))
    R+= 1e-4 * sp.eye(N)

    # define col covariances
    Cg = FreeFormCov(P)
    Cn = FreeFormCov(P)

    pdb.set_trace()

    # debug covarianec
    cov = Cov3KronSumLR(Cn = Cn, Cg = Cg, R = R, G = G, rank = 1)
    cov.setRandomParams()
    pdb.set_trace()
    cov.K()
    print ((cov.H_chol_debug()-cov.H_chol())**2).mean()<1e-9
    print ((cov.inv_debug()-cov.inv())**2).mean()<1e-9
    print (cov.logdet_debug()-cov.logdet())**2
    print (cov.logdet_grad_i_debug(0)-cov.logdet_grad_i(0))**2

