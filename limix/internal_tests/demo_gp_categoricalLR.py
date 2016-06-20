import sys
sys.path.insert(0, '../..')

from limix.core.mean.mean_base import MeanBase as lin_mean
from limix.core.covar import CategoricalLR
from limix.core.covar import FreeFormCov
from limix.core.covar import DiagonalCov
from limix.core.covar import KronCov
from limix.core.covar import SumCov
from limix.core.gp import GP

import pdb
import scipy as sp
import scipy.linalg as la
import time as TIME
import copy
import pylab as pl

sp.random.seed(1)

if __name__ == "__main__":

    # generate data
    N = 1000; S = 10
    Y = sp.randn(N, 1)
    G = 1. * (sp.rand(N, S)<0.2)
    G-= G.mean(0); G/= G.std(0); G /= sp.sqrt(S)
    #Ie = sp.rand(N)<0.5
    Ie = sp.arange(N) < (0.5*N)
    Cr = FreeFormCov(2, jitter=0.)
    covar = CategoricalLR(Cr, G, Ie)
    covar.setRandomParams()

    # build incomplete kronecker covar
    Rr = sp.dot(G, G.T)
    Rn = sp.eye(N)
    YM = sp.nan * sp.zeros((N,2))
    YM[Ie, 0] = Y[Ie, 0]
    YM[~Ie, 1] = Y[~Ie, 0]
    Iok = ~sp.isnan(YM).reshape(YM.size, order='F')
    covar1 = SumCov(KronCov(covar.Cr, Rr, Iok=Iok), KronCov(covar.Cn, Rn, Iok=Iok))

    if 0:
        # build complete kronecker covar
        covar1 = SumCov(KronCov(covar.Cr, Rr), KronCov(covar.Cn, Rn))
        # bridge kron with non-kron
        idxsM = sp.arange(2*N).reshape((N,2), order='F')
        idxs = sp.zeros(N, dtype=int)
        idxs[Ie]  = idxsM[Ie, 0]
        idxs[~Ie] = idxsM[~Ie, 1]
        print('K:', ((covar1.K()[idxs][:, idxs] - covar.K())**2).mean())

    # compare
    print('K:', ((covar1.K() - covar.K())**2).mean())
    print('Kiy:', ((covar.solve(Y)-covar1.solve(Y))**2).mean())
    print('logdet:', covar.logdet()-covar1.logdet())
    for i in range(covar.getNumberParams()):
        print('K_grad_%d:'%i, ((covar1.K_grad_i(i) - covar.K_grad_i(i))**2).mean())
        print('logdet_grad_%d:'%i, covar1.logdet_grad_i(i) - covar.logdet_grad_i(i))

    # define mean term
    W = 1. * (sp.rand(N, 2) < 0.2)
    mean = lin_mean(Y, W)

    # define gp
    params0 = covar.getParams().copy()
    gp = GP(covar=covar,mean=mean)
    gp.optimize()

    # define gp1
    covar.setParams(params0)
    gp1 = GP(covar=covar1,mean=mean)
    gp1.optimize()

    pdb.set_trace()


