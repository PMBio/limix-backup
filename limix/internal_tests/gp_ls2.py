import sys
sys.path.insert(0,'./../..')

from limix.core.mean.mean_base import MeanBase as lin_mean
from limix.core.covar import SQExpCov
from limix.core.covar import FixedCov
from limix.core.covar import SumCov
from limix.core.gp import GP
from limix.core.gp import GPLS
from limix.utils.preprocess import covar_rescale

import pdb
import scipy as sp
import scipy.linalg as LA
import time as TIME
import copy
import pylab as pl

if __name__ == "__main__":

    sp.random.seed(1)

    # generate data
    N = 1000 
    r = 5
    h2 = 0.30
    X1 = 1. * (sp.rand(N, r) < 0.2)
    X2 = 1. * (sp.rand(N, r) < 0.2)
    X3 = 1. * (sp.rand(N, r) < 0.2)
    K1 = covar_rescale(sp.dot(X1, X1.T))
    K2 = covar_rescale(sp.dot(X2, X2.T))
    K3 = covar_rescale(sp.dot(X3, X3.T))
    Y1 = sp.dot(X1, sp.randn(r, 1))
    Y2 = sp.dot(X2, sp.randn(r, 1))
    Y3 = sp.dot(X3, sp.randn(r, 1))
    Yn = sp.randn(N, 1)
    Y1*= sp.sqrt(h2 / 3 * Y1.var())
    Y2*= sp.sqrt(h2 / 3 * Y2.var())
    Y3*= sp.sqrt(h2 / 3 * Y3.var())
    Yn*= sp.sqrt((1 - h2) / Yn.var())
    Y  = Y1 + Y2 + Y3 + Yn

    # define mean term
    mean = lin_mean(Y)

    # define covariance matrices
    sign1 = FixedCov(K1)
    sign2 = FixedCov(K2)
    sign3 = FixedCov(K3)
    noise = FixedCov(sp.eye(N))
    covar  = SumCov(sign1, sign2, sign3, noise)
    covar._nIterMC = 200
    covar._reuse = False

    # define normal gp
    gp = GP(covar=covar, mean=mean)
    # define lin sys gp
    gpls = GPLS(Y, covar)
    sp.random.seed(1)
    gpls.covar.Z()

    if 0:
        # initialize params
        sign1.scale = 0.25
        sign2.scale = 0.25
        sign3.scale = 0.25
        noise.scale = 0.25
        gp.optimize(calc_ste=True)
        print 'params:'
        print sign1.scale, '+/-', sign1.scale_ste
        print sign2.scale, '+/-', sign2.scale_ste
        print sign3.scale, '+/-', sign3.scale_ste
        print noise.scale, '+/-', noise.scale_ste
        pdb.set_trace()

    # initialize params
    sign1.scale = 0.25
    sign2.scale = 0.25
    sign3.scale = 0.25
    noise.scale = 0.25
    print gpls.getParams()['covar']
    gpls.optimize(debug=True)
    print 'params:'
    print sign1.scale
    print sign2.scale
    print sign3.scale
    print noise.scale
    pdb.set_trace()

    # initialize params
    sign1.scale = 0.25
    sign2.scale = 0.25
    sign3.scale = 0.25
    noise.scale = 0.25
    # set reusage
    covar._reuse = True 
    gpls.optimize(debug=True)
    print 'params:'
    print sign1.scale
    print sign2.scale
    print sign3.scale
    print noise.scale
    pdb.set_trace()

