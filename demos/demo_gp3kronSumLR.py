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

    # define GP and optimize
    gp = GP3KronSumLR(Y = Y, Cg = Cg, Cn = Cn, R = R, G = G, rank = 1)
    gp.optimize()

