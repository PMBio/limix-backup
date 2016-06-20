import sys
sys.path.insert(0,'./../../..')

from limix.core.mean.mean_base import MeanBase as lin_mean
from limix.core.covar import FixedCov
from limix.core.covar import SumCov
from limix.core.gp import GP

import pdb
import scipy as sp
import scipy.linalg as LA
import time as TIME
import copy
import pylab as pl

sp.random.seed(1)

if __name__ == "__main__":

    # generate data
    N = 2000
    S = 100 # number of SNPs
    X = 1. * (sp.rand(N, S)<0.2)
    X-= X.mean(0); X/= X.std(0)
    K = sp.dot(X, X.T) / float(S)

    # Y = Xb + \psi
    # X is NxS
    # b is Sx1
    # \psi is noise (will be ~N(...))
    var_g = 0.2
    b = sp.randn(S, 1)
    Yg = sp.dot(X, b)
    Yn = sp.randn(N, 1)
    Yg*= sp.sqrt(var_g / Yg.var())
    Yn*= sp.sqrt((1. - var_g) / Yn.var())
    Y = Yg + Yn

    import ipdb
    ipdb.set_trace()

    # define mean term
    mean = lin_mean(Y)

    # define covariance matrices
    geno = FixedCov(K)
    noise = FixedCov(sp.eye(N))
    covar  = SumCov(geno, noise)

    # define gp
    gp = GP(covar=covar,mean=mean)
    # initialize params
    geno.scale = 0.5 
    noise.scale = 0.5
    # optimize
    gp.optimize(calc_ste=True)

    # print optimized values and standard errors
    print('scale of geno')
    print(geno.scale, '+/-', geno.scale_ste)
    print('scale of fixed')
    print(noise.scale, '+/-', noise.scale_ste)

