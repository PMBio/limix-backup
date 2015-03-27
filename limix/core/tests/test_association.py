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


def generate_data(N=100,S=1000,R=1000,P=2,h2_S=0.1,h2_R=0.0,maf=0.5):
    """
    generate SNPs, random effects and phenotypes

    args:
        N     number individuals
        S     number SNPs
        R     number random effects
        P     number phenotypes
        h2_S  heritability SNPs
        h2_R  heritability of random effects
        maf   minimum allele frequency

    returns:
        snps  [N x S] ndarray of SNPs
        X     [N x R] ndarray of random effects
        Y     [N x P] ndarray of phenotypes
    """
    
    assert h2_R>=0, "h2_R has to be greater or equal to zero"
    assert h2_S>=0, "h2_S has to be greater or equal to zero"
    assert h2_R+h2_S<=1, "h2_R + h2_S has to be smaller or equal to one"

    X = 1.*(SP.rand(N,R)<0.2)#random effects
    beta = SP.randn(R,P)
    Yconf = SP.dot(X,beta);
    Yconf*=SP.sqrt(h2_R/Yconf.var(0).mean())

    snps = 1.*(SP.rand(N,S)<maf)#SNP effects
    beta_snp = SP.randn(S,P)*10000
    Yg = SP.dot(snps,beta_snp);
    Yg*=SP.sqrt(h2_S/Yg.var(0).mean())


    Yn = SP.randn(N,P); Yn*=SP.sqrt((1-h2_S-h2_R))
    Y  = Yconf+Yg+Yn; Y/=Y.std(0)
    return snps, X, Y


if __name__ == "__main__":

    plot = "plot" in sys.argv

    # generate data
    N = 500; P = 2; S = 1000; R = 1000
    maf = 0.2
    h2_S = 0.3; h2_R = 0.2
    snps, X, Y = generate_data(N=N,S=S,R=R,P=P,h2_S=h2_S,h2_R=h2_R,maf=maf)

    # define mean term
    mu = mean(Y)

    # add any fixed effect including bias term
    F = 1.*(SP.rand(N,3)<0.2); A = SP.eye(P)
    F[:,0]=1.0
    mu.addFixedEffect(F=F,A=A)

    # add common fixed effect 
    F2 = 1.*(SP.rand(N,4)<0.2); A2 = SP.ones((1,P))    
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


    print "creating gp2kronSum object"
    XX = SP.dot(X,X.T)
    XX/= XX.diagonal().mean()
    gp = gp2kronSum(mu,Cg,Cn,XX)
    gp.setParams(params)

    if "ML" in sys.argv:
        print "ML estimation"
        gp.set_reml(False)
    else:
        print "REML estimation"
    
    print "optimization of GP parameters"
    start = TIME.time()
    conv,info = OPT.opt_hyper(gp,params,factr=1e3)
    print 'Reml GP:', TIME.time()-start
        

    print conv

    print "creating lmm for association using GP object" 
    assoc = lmm.LmmKronecker(gp=gp)
    
    #test snps
    print "testing SNPs with any effect"
    pv,LL_snps,LL_snps_0= assoc.test_snps(snps)
    
    if 0:
        import pylab as pl
        pl.ion()
        pl.hist(pv,30)
