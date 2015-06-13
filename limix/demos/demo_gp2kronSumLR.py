import scipy as sp
from limix.core.covar import Cov2KronSumLR
from limix.core.covar import FreeFormCov 
from limix.utils.preprocess import covar_rescale
import time
import copy
import pdb

if __name__=='__main__':

    # define row caoriance
    N = 200
    f = 10
    P = 3
    X = 1.*(sp.rand(N, f)<0.2)

    # define col covariances
    Cn = FreeFormCov(P)

    cov = Cov2KronSumLR(Cn = Cn, G = X, rank = 1)
    cov.setRandomParams()
    pdb.set_trace()
    print ((cov.inv_debug()-cov.inv())**2).mean()<1e-9
    print (cov.logdet_debug()-cov.logdet())**2
    print (cov.logdet_grad_i_debug(0)-cov.logdet_grad_i(0))**2

    pdb.set_trace()

