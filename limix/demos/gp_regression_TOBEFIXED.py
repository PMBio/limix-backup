import sys
sys.path.insert(0,'./../../..')

from limix.core.mean.mean_base import mean_base as lin_mean
from limix.core.covar.sqexp import sqexp
from limix.core.covar.fixed import fixed
from limix.core.covar.combinators import sumcov
from limix.core.gp.gp_base_new import gp as gp_base

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
    Y-= Y.mean(0)
    Y/= Y.std(0)

    Xstar = sp.linspace(0,2,1000)[:,sp.newaxis]

    # define mean term
    F = 1.*(sp.rand(N,2)<0.2)
    mean = lin_mean(Y,F)

    ipdb.set_trace()

    # define covariance matrices
    covar1 = sqexp(X,Xstar=Xstar)
    covar2 = fixed(sp.eye(N))
    covar  = sumcov(covar1,covar2)

    if 0:
        # cheack gradient for covariances
        for i in range(10):
            covar1.setRandomParams()
            covar1.test_grad()
            covar2.setRandomParams()
            covar2.test_grad()
            covar.setRandomParams()
            covar.test_grad()

    # define gp
    gp = gp_base(covar=covar,mean=mean)
    covar.setRandomParams()
    print gp.LML()
    print gp.LML_grad()

    gp.predict()

    ipdb.set_trace()
    gp.checkGradient(fun='yKiy')
    gp.checkGradient(fun='yKiFb')
    gp.checkGradient(fun='LML')

    gp.covar.getCovariance(0).scale = 1e-4
    gp.covar.getCovariance(0).length = 1
    gp.covar.getCovariance(1).scale = 1 
    gp.optimize(calc_ste=True)

    # print optimized values and standard errors
    print 'weights of fixed effects'
    print mean.b
    print '+/-',mean.b_ste
    print 'scale of sqexp'
    print covar1.scale
    print '+/-',covar1.scale_ste
    print 'length of sqexp'
    print covar1.length
    print '+/-',covar1.length_ste
    print 'scale of fixed'
    print covar2.scale
    print '+/-',covar2.scale_ste

    ipdb.set_trace()
    Ystar = gp.predict()
    pl.plot(X.ravel(),Y.ravel(),'xk')
    pl.plot(Xstar.ravel(),Ystar.ravel(),'FireBrick',lw=2)

