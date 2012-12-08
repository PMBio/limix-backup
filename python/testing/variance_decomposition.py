import scipy as SP
import h5py
import sys
sys.path.append('./../../build/src/interfaces/python')

import pylab as PL
import limix
import numpy
import numpy.random as random


if __name__ == "__main__":
    random.seed(1)
    
    N = 10
    S = 1000
    snps = SP.randn(N,S)
    y1    = 0.4*snps[:,10][:,SP.newaxis]
    y1   += 0.1*SP.randn(N,1)
    y2    = 0.8*snps[:,10][:,SP.newaxis]
    y2   += 0.2*SP.randn(N,1)

    Y = SP.concatenate([y1,y2],axis=0)
    X = SP.concatenate([snps,snps],axis=0)
    T = SP.zeros([Y.shape[0],1])
    T[0:N] = 0
    T[N::] = 1


    #get variance decomposition going
    K = SP.dot(X,X.T)
    Keye = SP.eye(N)
    Keye = SP.concatenate([Keye,Keye],axis=0)
    Kgeno  = SP.dot(Keye,Keye.T)
    
    v = limix.CVarianceDecomposition(Y,T)
    
    #estimate approxiamtely the variances
    [vm0,vm1] = limix.CVarianceDecomposition.aestimateHeritability(Y,SP.ones([Y.shape[0],1]),K)
    
    if 1:
        v.addTerm(K,limix.CVarianceDecomposition.categorial,vm0,True)
        v.addTerm(Kgeno,limix.CVarianceDecomposition.categorial,vm1,False)
        v0 = v.getTerm(0)
        v1 = v.getTerm(1)
    else:
        v0 = limix.CCategorialTraitVarianceTerm(K,T,True)
        v1 = limix.CCategorialTraitVarianceTerm(Kgeno,T,vm1,False)
        v0.setVinitMarginal(vm0*SP.ones([2]))
        v.addTerm(v0)
        v.addTerm(v1)
    v.train()
    
    o = v.getOpt()
    gp = v.getGP()
    
    
    print gp.LML()