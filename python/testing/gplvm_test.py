# Copyright(c) 2014, The LIMIX developers (Christoph Lippert, Paolo Francesco Casale, Oliver Stegle)
# All rights reserved.
#
# LIMIX is provided under a 2-clause BSD license.
# See license.txt for the complete license.

import sys
sys.path.append('./../../release.darwin/interfaces/python')


import limix
import scipy as SP
import pdb
import time
import scipy.linalg as linalg

def PCA(Y, components):
    """run PCA, retrieving the first (components) principle components
    return [s0, eig, w0]
    s0: factors
    w0: weights
    """
    sv = linalg.svd(Y, full_matrices=0);
    [s0, w0] = [sv[0][:, 0:components], SP.dot(SP.diag(sv[1]), sv[2]).T[:, 0:components]]
    v = s0.std(axis=0)
    s0 /= v;
    w0 *= v;
    return [s0, w0]


SP.random.seed(1)

#1. simulate data from a linear PCA model
N = 100
K = 5 
D = 100

SP.random.seed(1)
S = SP.random.randn(N,K)
W = SP.random.randn(D,K)

Y = SP.dot(W,S.T).T
Y+= 0.1*SP.random.randn(N,D)

X0 = SP.random.randn(N,K)
X0 = PCA(Y,K)[0]

#starting params
covar_params = SP.array([1.0])
lik_params = SP.array([0.1])

#GPMIX:
covar  = limix.CCovLinearISO(K)
ll  = limix.CLikNormalIso()
#create hyperparm     
hyperparams = limix.CGPHyperParams()
hyperparams['covar'] = covar_params
hyperparams['lik'] = lik_params
hyperparams['X']   = X0
#cretae GP
gp=limix.CGPbase(covar,ll)
#set data
gp.setY(Y)
gp.setX(X0)
lml0 = gp.LML(hyperparams)
dlml0 = gp.LMLgrad(hyperparams)

#optimization
lml0 = gp.LML()
dlml0 = gp.LMLgrad(hyperparams)

gpopt = limix.CGPopt(gp)
t2 = time.time()
gpopt.opt()
t3 = time.time()


