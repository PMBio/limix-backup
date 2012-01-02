import sys
sys.path.append('./..')
sys.path.append('./../../../pygp')


import gpmix
import pygp.covar.linear as lin
import pygp.likelihood as lik
from pygp.gp import gp_base,gplvm,gplvm_ard
from pygp.covar import linear,se, noise, combinators
#import pygp.covar.gradcheck as GC
import pygp.covar.combinators as comb
import pygp.optimize.optimize_base as opt
import scipy as SP
import pdb
import time

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

#use "standard PCA"
[Spca,Wpca] = gplvm.PCA(Y,K)
X0 = Spca
#X0 = SP.random.randn(N,K)

#starting params
covar_params = SP.random.randn(1,1)
lik_params = SP.log(SP.array([0.1]))

#pygp: OLD
covariance = linear.LinearCFISO(n_dimensions=K)
#standard Gaussian noise
likelihood = lik.GaussLikISO()
hyperparams_ = {}
hyperparams_['lik'] = lik_params.copy()
hyperparams_['covar'] = covar_params.copy()
hyperparams_['x'] = X0.copy()

g = gplvm.GPLVM(covar_func=covariance,likelihood=likelihood,x=X0,y=Y,gplvm_dimensions=SP.arange(X0.shape[1]))
lml0_ = g.LML(hyperparams_)
t0 = time.time()
[opt_hyperparams_,opt_lml2] = opt.opt_hyper(g,hyperparams_,gradcheck=False)
t1 = time.time()
lml_ = g.LML(opt_hyperparams_)
dlml_ = g.LMLgrad(opt_hyperparams_)


#GPMIX:
covar  = gpmix.CCovLinearISO(K)
ll  = gpmix.CLikNormalIso()
#create hyperparm     
hyperparams = gpmix.CGPHyperParams()
hyperparams['covar'] = covar_params
hyperparams['lik'] = lik_params
hyperparams['X']   = X0
#cretae GP
gp=gpmix.CGPbase(covar,ll)
#set data
gp.setY(Y)
gp.setX(X0)
lml0 = gp.LML(hyperparams)
dlml0 = gp.LMLgrad(hyperparams)

#optimization
lml0 = gp.LML()
dlml0 = gp.LMLgrad(hyperparams)

gpopt = gpmix.CGPopt(gp)
t2 = time.time()
gpopt.opt()
t3 = time.time()

opt_params_GP = gp.getParams()
opt_params_True = gpopt.getOptParams()

lml = gp.LML()
dlml = gp.LMLgrad()

print "optimization timing:"
print (t1-t0)
print (t3-t2)

