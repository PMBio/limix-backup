import sys
sys.path.append('./..')
sys.path.append('./../../../pygp')


import gpmix
import pygp.covar.linear as lin
import pygp.likelihood as lik
from pygp.gp import gp_base,gplvm,kronecker_gplvm,gplvm_ard
from pygp.covar import linear,se, noise, combinators, fixed
#import pygp.covar.gradcheck as GC
import pygp.covar.combinators as comb
import pygp.optimize.optimize_base as opt
import scipy as SP
import pdb
import time
import logging as LG


if __name__ == '__main__':
    LG.basicConfig(level=LG.INFO)
    #
    #1. simulate data from a linear PCA model
    #note, these data are truely independent across rows, so the whole gplvm with kronecker is a bit pointless....
    N = 50
    K = 5
    D = 20

    SP.random.seed(1)
    S = SP.random.randn(N,K)
    W = SP.random.randn(D,K)

    Y = SP.dot(W,S.T).T
    Y += 0.1*SP.random.randn(N,D)

    X0= SP.random.randn(N,K)

    Kc = SP.eye(D)
    covariance_c = fixed.FixedCF(Kc)
    covariance_r = linear.LinearCFISO(n_dimensions=K)
    
    likelihood = lik.GaussLikISO()

    hyperparams = {}
    hyperparams['lik'] = SP.log([0.42])
    hyperparams['covar_r'] = SP.log([1.0])
    hyperparams['covar_c'] = SP.log([1.0])
    hyperparams['x_r'] = X0

    kgp = kronecker_gplvm.KroneckerGPLVM(covar_func_r=covariance_r,covar_func_c=covariance_c,likelihood=likelihood)
    kgp.setData(x_r=X0,y=Y)
    gradcheck=True
    [hyperparams_o,opt_lml_o] = opt.opt_hyper(kgp,hyperparams,gradcheck=gradcheck)
    
    