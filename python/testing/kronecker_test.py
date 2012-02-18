import sys
sys.path.append('./..')
sys.path.append('./../../../pygp')


import gpmix
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
    N = 100
    Kr = 3
    Kc = 2
    D = 20 

    SP.random.seed(1)
    S = SP.random.randn(N,Kr)
    W = SP.random.randn(D,Kr)

    Y = SP.dot(W,S.T).T
    Y += 0.1*SP.random.randn(N,D)

    X0r= SP.random.randn(N,Kr)
    X0c= SP.random.randn(D,Kc)
    

    ard = True
    if ard:
        covar_r = SP.zeros([Kr])+0.2*SP.random.randn(Kr)
        covar_c = SP.zeros([Kc])+0.2*SP.random.randn(Kc)
        lik = SP.log([0.1])
    else:    
        covar_r = SP.log([0.5])
        covar_c = SP.log([0.3])
        lik = SP.log([0.1])
        
    if ard:
        covariance_c_ = linear.LinearCF(n_dimensions=Kc)
        covariance_r_ = linear.LinearCF(n_dimensions=Kr)
    else:
        covariance_c_ = linear.LinearCFISO(n_dimensions=Kc)
        covariance_r_ = linear.LinearCFISO(n_dimensions=Kr)
    
    likelihood_ = LIK.GaussLikISO()

    hyperparams = {}
    hyperparams['covar_r'] = covar_r
    hyperparams['covar_c'] = covar_c
    hyperparams['lik'] = lik
    hyperparams['x_r'] = X0r
    hyperparams['x_c'] = X0c
    
    
    kgp = kronecker_gplvm.KroneckerGPLVM(covar_func_r=covariance_r_,covar_func_c=covariance_c_,likelihood=likelihood_)
    kgp.setData(x_r=X0r,x_c=X0c,y=Y)
    
    #gradcheck=True
    #[hyperparams_o,opt_lml_o] = opt.opt_hyper(kgp,hyperparams,gradcheck=gradcheck)
    


    #gpmix
    if ard:
        covariance_r = gpmix.CCovLinearARD(Kr)
        covariance_c = gpmix.CCovLinearARD(Kc)
    else:
        covariance_r = gpmix.CCovLinearISO(Kr)
        covariance_c = gpmix.CCovLinearISO(Kc)
    gp=gpmix.CGPkronecker(covariance_r,covariance_c)
    gp.setY(Y);
    gp.setX_r(X0r);
    gp.setX_c(X0c);
    params = gpmix.CGPHyperParams();
    params['covar_r'] = covar_r
    params['covar_c'] = covar_c
    params['lik'] = lik
    params['X_r'] = X0r
    params['X_c'] = X0c
    
    gp.setParams(params)
    
    #1. compare K
    print ((covariance_r_.K(hyperparams['covar_r'],X0r)-covariance_r.K())**2).max()
    print ((covariance_c_.K(hyperparams['covar_c'],X0c)-covariance_c.K())**2).max()
    
    
    LML = gp.LML()
    LML_ = kgp.LML(hyperparams)
    
    LMLgrad = gp.LMLgrad()
    LMLgrad_ = kgp.LMLgrad(hyperparams)
    

    print SP.absolute(LML-LML_)
    print SP.absolute(LMLgrad['covar_r']-LMLgrad_['covar_r']).max()
    print SP.absolute(LMLgrad['covar_c']-LMLgrad_['covar_c']).max()
    print SP.absolute(LMLgrad['lik']-LMLgrad_['lik']).max()
    print SP.absolute(LMLgrad['X_r']-LMLgrad_['x_r']).max()
    #print SP.absolute(LMLgrad['X_c']-LMLgrad_['x_c']).max()
    
    gpopt = gpmix.CGPopt(gp)
    print "gradcheck"
    print gpopt.gradCheck()
    
    #gpopt.opt()
    
    t1 = time.time()
    SP.random.seed(1)
    for i in xrange(100):
        print i
        hyperparams['covar_r'][0] = SP.random.randn()
        kgp.LMLgrad(hyperparams)
    t2 = time.time()
    

    t3 = time.time()
    SP.random.seed(1)
    for i in xrange(100):
        print i
        covar_r[0] = SP.random.randn()
        params['covar_r'] = covar_r
        gp.setParams(params)
        gp.LMLgrad()
    t4 = time.time()
    
    
