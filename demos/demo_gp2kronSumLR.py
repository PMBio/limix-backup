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
    N = 1000
    f = 10
    P = 3
    X = 1.*(sp.rand(N, f)<0.2)

    # define col covariances
    Cn = FreeFormCov(P)
    Cn.setRandomParams()

    # define fixed effects and pheno
    F = 1.*(sp.rand(N,2)<0.5)
    A = sp.eye(P)
    Y = sp.randn(N, P)

    # define gp and optimize
    gp = GP2KronSumLR(Y = Y, F = F, A = A, Cn = Cn, G = X)
    gp.optimize()

