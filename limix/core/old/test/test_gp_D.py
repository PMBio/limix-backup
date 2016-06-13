import sys

try:
    sys.path.insert(0,'./../../..')
    sys.path.insert(0,'./../../../../../mksum/mtSet')
    from mtSet.pycore.gp import gp2kronSum as gp2kronSumMtSet
    import mtSet.pycore.mean as MEAN
    mtSet_present = True
except:
    print("no mtSet found in path")
    mtSet_present = False

from limix.core.mean import mean
from limix.core.gp import gp2kronSum as gp2kronSum

from limix.core.covar import freeform 
import limix.core.optimize.optimize_bfgs as OPT

import ipdb
import scipy as SP
import scipy.linalg as LA 
import time as TIME
import copy


if __name__ == "__main__":

    # generate data
    h2 = 0.3
    N = 500; P = 3; S = 1000
    X = 1.*(SP.rand(N,S)<0.2)
    beta = SP.randn(S,P)
    Yg = SP.dot(X,beta); Yg*=SP.sqrt(h2/Yg.var(0).mean())
    Yn = SP.randn(N,P); Yn*=SP.sqrt((1-h2)/Yn.var(0).mean())
    Y  = Yg+Yn; Y-=Y.mean(0); Y/=Y.std(0)
    XX = SP.dot(X,X.T)
    XX/= XX.diagonal().mean()
    Xr = 1.*(SP.rand(N,10)<0.2)
    Xr*= SP.sqrt(N/(Xr**2).sum())

    # define mean term
    mu = mean(Y)

    # add first fixed effect
    F = 1.*(SP.rand(N,2)<0.2); A = SP.eye(P)

    # add second fixed effect
    F2 = 1.*(SP.rand(N,3)<0.2); A2 = SP.ones((1,P))
    
    mu.addFixedEffect(F=F,A=A)
    mu.addFixedEffect(F=F2,A=A2)

    if mtSet_present:
        mu1 = MEAN.mean(Y)
        mu1.addFixedEffect(F=F,A=A)
        mu1.addFixedEffect(F=F2,A=A2)
    

    # define covariance matrices
    Cg = freeform(P)
    Cn = freeform(P)
    Cg1 = freeform(P)
    Cn1 = freeform(P)
    
    ipdb.set_trace()

    if 1:
        # compare with mtSet implementation
        params = {}
        params['Cg']   = SP.randn(int(0.5*P*(P+1)))
        params['Cn']   = SP.randn(int(0.5*P*(P+1)))


        print("check gradient with gp2kronSum")
        gp = gp2kronSum(mu,Cg,Cn,XX)
        gp.setParams(params)

        if 0:
            gp.set_reml(False)

        print("test optimization")
        start = TIME.time()
        conv,info = OPT.opt_hyper(gp,params,factr=1e3)
        print(('Reml GP:', TIME.time()-start))
        
        if mtSet_present:
            params1 = copy.copy(params)
            params1['mean'] = SP.zeros(mu1.getParams().shape[0])
            gp1 = gp2kronSumMtSet(mu1,Cg1,Cn1,XX)
            gp1.setParams(params1)
            start = TIME.time()
            conv1,info = OPT.opt_hyper(gp1,params1,factr=1e3)
            print(('Old GP:', TIME.time()-start))

        print(conv)

        ipdb.set_trace()

    if 1:
        # no fixed 
        mu = mean(Y)
        params = {}
        params['Cg']   = SP.randn(int(0.5*P*(P+1)))
        params['Cn']   = SP.randn(int(0.5*P*(P+1)))

        ipdb.set_trace()
        gp = gp2kronSum(mu,Cg,Cn,XX)
        gp.setParams(params)
        gp.LML()
        gp.LMLgrad()

