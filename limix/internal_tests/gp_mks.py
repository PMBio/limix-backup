import sys
sys.path.insert(0,'./../..')

from limix.core.covar import CovMultiKronSum
from limix.core.covar import FreeFormCov 
from limix.core.covar import SumCov 
from limix.core.covar import KronCov 
from limix.utils.preprocess import covar_rescale

import ipdb
import scipy as sp
import scipy.linalg as LA
import time as TIME
import copy
import pylab as pl

sp.random.seed(1)

def generate_data(N, P, f, n_terms):
    Y = sp.zeros((N, P))
    R = []; C = []
    for term_i in range(n_terms):
        if term_i==n_terms-1:
            X = sp.eye(N)
        else:
            X = sp.zeros((N, N))
            X[:, :f] = 1.*(sp.rand(N, f)<0.2)
        W = sp.randn(P, P)
        Z = sp.randn(N, P)
        Z = sp.dot(X, sp.dot(Z, W.T))
        Z*= sp.sqrt(1. / (n_terms * Z.var(0).mean()))
        Y+= Z 
        _R = covar_rescale(sp.dot(X,X.T))
        _R+= 1e-4 * sp.eye(N)
        R.append(_R)
        C.append(FreeFormCov(P))
    Y -= Y.mean(0)
    Y /= Y.std(0)
    return Y, C, R
        
if __name__ == "__main__":

    # generate data
    N = 1000
    P = 2 
    f = 10
    n_terms = 3

    Y, C, R = generate_data(N, P, f, n_terms)

    # standard sum of Kroneckers
    covar0 = SumCov(*[KronCov(C[i], R[i]) for i in range(len(C))])
    covar0.setRandomParams()

    # specialized sum of Kroneckers
    covar = CovMultiKronSum(C, R) 

    ipdb.set_trace()

    if 1:
        # basic checks
        print ((covar.K()-covar0.K())**2).mean()
        print ((covar.K_grad_i(0)-covar0.K_grad_i(0))**2).mean()
        #print ((covar.K_hess_i_j(0, 1)()-covar0.K_hess_i_j(0, 0))**2).mean()
        ipdb.set_trace()


