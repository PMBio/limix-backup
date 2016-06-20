import sys
sys.path.insert(0,'./../../..')

from limix.core.mean.mean_base import MeanBase as lin_mean
from limix.core.covar import SQExpCov
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
    N = 400
    X = sp.linspace(0,2,N)[:,sp.newaxis]
    v_noise = 0.01
    Y = sp.sin(X) + sp.sqrt(v_noise) * sp.randn(N, 1)

    # for out-of-sample preditions
    Xstar = sp.linspace(0,2,1000)[:,sp.newaxis]

    # define mean term
    W = 1. * (sp.rand(N, 2) < 0.2)
    mean = lin_mean(Y, W)

    # define covariance matrices
    sqexp = SQExpCov(X, Xstar = Xstar)
    noise = FixedCov(sp.eye(N))
    covar  = SumCov(sqexp, noise)

    # define gp
    gp = GP(covar=covar,mean=mean)
    # initialize params
    sqexp.scale = 1e-4
    sqexp.length = 1
    noise.scale = Y.var()
    # optimize
    gp.optimize(calc_ste=True)
    # predict out-of-sample
    Ystar = gp.predict()

    # print optimized values and standard errors
    print('weights of fixed effects')
    print(mean.b[0, 0], '+/-', mean.b_ste[0, 0])
    print(mean.b[1, 0], '+/-', mean.b_ste[1, 0])
    print('scale of sqexp')
    print(sqexp.scale, '+/-', sqexp.scale_ste)
    print('length of sqexp')
    print(sqexp.length, '+/-', sqexp.length_ste)
    print('scale of fixed')
    print(noise.scale, '+/-', noise.scale_ste)

    # plot
    pl.subplot(111)
    pl.title('GP regression with SQExp')
    pl.plot(X.ravel(),Y.ravel(), 'xk', label = 'Data points')
    pl.plot(Xstar.ravel(),Ystar.ravel(),'FireBrick',lw=2, label = 'GP')
    pl.xlabel('x')
    pl.ylabel('y')
    pl.legend(loc = 4)
    pl.tight_layout()
    pl.show()
