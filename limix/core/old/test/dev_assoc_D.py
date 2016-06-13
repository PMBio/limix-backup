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
from limix.core.mean.mean_efficient import Mean
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

    plot = "plot" in sys.argv

    # generate data
    h2 = 0.999
    N = 500; P = 2; S = 1000
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
    F2 = 1.*(SP.rand(N,4)<0.2); A2 = SP.ones((1,P))
    
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


    print("check gradient with gp2kronSum")
    gp = gp2kronSum(mu,Cg,Cn,XX)
    gp.setParams(params)

    if "ML" in sys.argv:
        gp.set_reml(False)
    if 1:#optimizwe
        print("test optimization")
        start = TIME.time()
        conv,info = OPT.opt_hyper(gp,params,factr=1e3)
        print(('Reml GP:', TIME.time()-start))
        

        print(conv)

    #association
    assoc = lmm.LmmKronecker(gp=gp)
    if 1:
        #signal snps
        pv,LL_snps,LL_snps_0= assoc.test_snps(snps, identity_trick = True)
        #null snps
        pv0,n_LL_snps0,LL_snps0_0 = assoc.test_snps(snps_0, identity_trick = True)
    
    if plot:
        import pylab as pl
        pl.ion()
        #pl.figure(); pl.hist(pv,50)
        #pl.figure(); pl.hist(pv0,50)
        i_pv = pv.argsort()
        i_pv0 = pv0.argsort()
        pl.figure()
        pl.plot(-SP.log(pv[i_pv]),-SP.log(pv0[i_pv0]),'.')
        pl.plot([0,8],[0,8])

        pv0_,n_LL_snps0_,LL_snps0_0_ = assoc.test_snps(snps_0, identity_trick = True)
        pl.figure(); pl.plot(-SP.log(pv0_),-SP.log(pv0),'.')
        pl.figure(); pl.plot(-SP.log(pv0_),-SP.log(pv0),'.')

    if 1:
        lml0 = assoc._gp.LMLdebug()
        lml1 = assoc._gp.LML()  
        gp_ = assoc._gp
        var_expl1,beta1 = assoc._gp.mean.var_explained()
        

        
        var_expl1,beta1 = assoc._gp.mean.var_explained()
        mu_ = Mean(Y)
        mu_.addFixedEffect(F=F,A=A)
        mu_.addFixedEffect(F=F2,A=A2)
        gp_.setMean(mu_)
        lml2 = gp_.LML()
        var_expl2,beta_hat, beta_hat_any = assoc._gp.mean.var_explained()

        mu__ = mean(Y,identity_trick=True)
        mu__.addFixedEffect(F=F,A=A)
        mu__.addFixedEffect(F=F2,A=A2)
        gp_.setMean(mu__)
        var_expl3,beta3 = assoc._gp.mean.var_explained()
        print(lml0)#complete garbage
        print(lml1)
        print(lml2)

    if 0:
        #forward selection step
        assoc.addFixedEffect(F=snps[:,i_pv[0]:(i_pv[0]+1)],A=None)
        
        if 1:#optimize
            print("test optimization")
            start = TIME.time()
            conv,info = OPT.opt_hyper(assoc._gp,params,factr=1e3)
            print(('Reml GP:', TIME.time()-start))
        

        print(conv)
        pv_forw,LL_snps_forw,LL_snps_0_forw = assoc.test_snps(snps,identity_trick = True)
        if plot:
            pl.figure()
            pl.plot(-SP.log(pv_forw),-SP.log(pv),'.')
            pl.plot([0,8],[0,8])


