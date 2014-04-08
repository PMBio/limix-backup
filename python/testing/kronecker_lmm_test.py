# Copyright(c) 2014, The LIMIX developers (Christoph Lippert, Paolo Francesco Casale, Oliver Stegle)
# All rights reserved.
#
# LIMIX is provided under a 2-clause BSD license.
# See license.txt for the complete license.

import sys
sys.path.append('./..')
sys.path.append('./../../../pygp')


import limix
import pygp.covar.linear as lin
import pygp.likelihood as LIK
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

    if 1:
        N = 100
        D = 20
        NS = 100

        SP.random.seed(1)
        ir = SP.random.permutation(NS)[0]

        S = SP.random.randn(N,NS)
        W = SP.random.randn(1,D)

        Y = SP.dot(S[:,ir:ir+1],W)
        Y += 0.1*SP.random.randn(N,D)


    if 1:
        covar_c = limix.CFixedCF(SP.eye(D))
        covar_r = limix.CFixedCF(SP.eye(N))
        Xr      = SP.zeros([N,0])
        Xc      = SP.zeros([D,0])

        gp = limix.CGPkronecker(covar_r,covar_c)
        gp.setX_r(Xr)
        gp.setX_c(Xc)
        gp.setY(Y)
        
        params = limix.CGPHyperParams()
        params["covar_r"] = SP.zeros([covar_r.getNumberParams()])
        params["covar_c"] = SP.zeros([covar_c.getNumberParams()])
        params["lik"] = SP.log([0.1,0.1])
    
        gp.setParams(params)
        opt=limix.CGPopt(gp)

        opt.opt()
        #opt.gradCheck()

    if 0:
        opt_params = limix.CGPHyperParams()
        opt_params["lik"] = params["lik"]
        gp.setParams(opt_params)
        pdb.set_trace()

        #lmm object
        lmm = limix.CGPLMM(gp)
        lmm.setSNPs(S)
        lmm.setCovs(SP.ones([N,1]))
        lmm.setPheno(Y)
        A = SP.ones([2,D])
        A0 = SP.ones([1,D])
        lmm.setA(A)
        lmm.setA0(A0)
        lmm.process()
        pv= lmm.getPv()
