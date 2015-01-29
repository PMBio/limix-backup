import sys

try:
    sys.path.insert(0,'./../../..')
    sys.path.insert(0,'./../../../../../mksum/mtSet')
    from mtSet.pycore.gp import gp2kronSum as gp2kronSumMtSet
    import mtSet.pycore.mean as MEAN
    mtSet_present = True
except:
    print "no mtSet found in path"
    mtSet_present = False

from limix.core.mean import mean
from limix.core.gp import gp2kronSum as gp2kronSum

from limix.core.covar import freeform 
import limix.core.optimize.optimize_bfgs as OPT
import limix.core.association.lmm_kronecker as lmm
import ipdb
import scipy as SP
import scipy.linalg as LA 
import time as TIME
import copy




if __name__ == "__main__":

    plot = True

    # generate data
    h2 = 0.999
    N = 500; P = 1; S = 1000
    X = 1.*(SP.rand(N,S)<0.2)
    snps = 1.*(SP.rand(N,S)<0.2)
    snps_0 = 1.*(SP.rand(N,S)<0.2)
    beta = SP.randn(S,P)
    beta_snp = SP.randn(S,P)*10000
    
    #beta_snp[(SP.rand(beta_snp.shape[0],beta_snp.shape[1])<0.1)]=0.0

    Yg = SP.dot(X,beta);
    Yg += SP.dot(snps,beta_snp);
    Yg*=SP.sqrt(h2/Yg.var(0).mean())
    Yn = SP.randn(N,P); Yn*=SP.sqrt((1-h2))
    Y  = Yg+Yn; Y-=Y.mean(0); Y/=Y.std(0)
    XX = SP.dot(X,X.T)
    XX/= XX.diagonal().mean()
    Xr = 1.*(SP.rand(N,10)<0.2)
    Xr*= SP.sqrt(N/(Xr**2).sum())

    # define mean term
    mu = mean(Y)

    # add first fixed effect
    F = 1.*(SP.rand(N,3)<0.2); A = SP.eye(P)
    F[:,0]=1.0

    # add second fixed effect
    F2 = 1.*(SP.rand(N,3)<0.2); A2 = SP.ones((1,P))
    
    mu.addFixedEffect(F=F,A=A)
    mu.addFixedEffect(F=F2,A=A2)
    

    # define covariance matrices
    Cg = freeform(P)
    Cn = freeform(P)
    Cg1 = freeform(P)
    Cn1 = freeform(P)
    
    # compare with mtSet implementation
    params = {}
    params['Cg']   = SP.randn(int(0.5*P*(P+1)))
    params['Cn']   = SP.randn(int(0.5*P*(P+1)))


    print "check gradient with gp2kronSum"
    gp = gp2kronSum(mu,Cg,Cn,XX)
    gp.setParams(params)

    if "ML" in sys.argv:
        gp.set_reml(False)

    print "test optimization"
    start = TIME.time()
    conv,info = OPT.opt_hyper(gp,params,factr=1e3)
    print 'Reml GP:', TIME.time()-start
        

    print conv

    #association
    assoc = lmm.LmmKronecker(gp=gp)
    #signal snps
    pv,LL_snps,LL_snps_0 = assoc.test_snps(snps)
    #null snps
    pv0,n_LL_snps0,LL_snps0_0 = assoc.test_snps(snps_0)
    
    if plot:
        import pylab as pl
        pl.ion()
        pl.figure(); pl.hist(pv,50)
        pl.figure(); pl.hist(pv0,50)
        i_pv = pv.argsort()
        i_pv0 = pv0.argsort()
        pl.figure()
        pl.plot(-SP.log(pv[i_pv]),-SP.log(pv0[i_pv0]),'.')
        pl.plot([0,8],[0,8])

    if 1:
        #forward selection step
        assoc.addFixedEffect(F=snps[:,i_pv[0]:(i_pv[0]+1)],A=None)
        pv_forw,LL_snps_forw,LL_snps_0_forw = assoc.test_snps(snps)
        if plot:
            pl.figure()
            pl.plot(-SP.log(pv_forw),-SP.log(pv),'.')
            pl.plot([0,8],[0,8])