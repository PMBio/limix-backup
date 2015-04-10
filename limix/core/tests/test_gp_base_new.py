import sys
sys.path.insert(0,'./../../..')

from limix.core.mean.mean_base import mean_base as lin_mean
from limix.core.covar.sqexp import sqexp
from limix.core.covar.fixed import fixed 
from limix.core.covar.combinators import sumcov 
from limix.core.gp.gp_base_new import gp

import ipdb
import scipy as sp
import scipy.linalg as LA 
import time as TIME
import copy
import pylab as pl
pl.ion()


if __name__ == "__main__":

    # generate data
    N = 400
    s_x = 0.05
    s_y = 0.1
    X = (sp.linspace(0,2,N)+s_x*sp.randn(N))[:,sp.newaxis]
    Y = sp.sin(X)+s_y*sp.randn(N,1)
    #pl.plot(x,y,'x')

    # define mean term
    F = 1.*(sp.rand(N,2)<0.2)
    mu = lin_mean(Y,F)

    # define covariance matrices
    covar1 = sqexp(X)
    covar2 = fixed(sp.eye(N))
    covar  = sumcov(covar1,covar2)

    ipdb.set_trace()

    # cheack gradient for covariances
    for i in range(10): 
        covar1.setRandomParams()
        covar1.test_grad()
        covar2.setRandomParams()
        covar2.test_grad()
        covar.setRandomParams()
        covar.test_grad()

    ipdb.set_trace()
    

