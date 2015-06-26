import sys
import os
import unittest
import ipdb
import scipy as sp
import scipy.linalg as la
from limix.utils.preprocess import covar_rescale
import limix


if __name__ == '__main__':

    # generate data
    N = 500
    f = 10
    P = 3
    K = 2
    Y = sp.randn(N, P)
    G = 1.*(sp.rand(N, f)<0.2)
    X = 1.*(sp.rand(N, f)<0.2)
    R = covar_rescale(sp.dot(X,X.T))
    R+= 1e-4 * sp.eye(N)
    S_R, U_R = la.eigh(R)
    F = sp.rand(N, K)

    ipdb.set_trace()

    # mtSet not PC
    #setTest = limix.MTSet(Y=Y, F=F)
    setTest = limix.MTSet(Y=Y, S_R=S_R, U_R=U_R)
    nullMTInfo = setTest.fitNull(cache=False)
    nullSTInfo = setTest.fitNullTraitByTrait(cache=False)

    optInfo = setTest.optimize(G)
    optInfo = setTest.optimizeTraitByTrait(G)

