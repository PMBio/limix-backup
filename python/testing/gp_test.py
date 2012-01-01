import sys
sys.path.append('./..')
sys.path.append('./../../../pygp')


import gpmix
import pygp.covar.linear as lin
import pygp.likelihood as lik
import pygp.gp.gp_base as GP
import pygp.covar.se as se
#import pygp.covar.gradcheck as GC
import pygp.covar.combinators as comb
import pygp.optimize.optimize_base as opt
import scipy as SP
import pdb
import time



n_dimensions=3
n_samples = 500
X = SP.randn(n_samples,n_dimensions)
y = SP.dot(X,SP.randn(n_dimensions,1))
y += 0.1*SP.randn(y.shape[0],y.shape[1])

covar_params = SP.random.randn(n_dimensions+1)
lik_params = SP.random.randn(1)

t0 = time.time()
#pygp: OLD
covar_ = se.SqexpCFARD(n_dimensions)
ll_ = lik.GaussLikISO()
hyperparams_ = {'covar':covar_params,'lik':lik_params}
gp_ = GP.GP(covar_,likelihood=ll_,x=X,y=y)
lml_ = gp_.LML(hyperparams_)
dlml_ = gp_.LMLgrad(hyperparams_)
#optimize using pygp:
opt_params_ = opt.opt_hyper(gp_,hyperparams_)[0]
lmlo_ = gp_.LML(opt_params_)
t1 = time.time()



#GPMIX:
covar  = gpmix.CCovSqexpARD(n_dimensions)
ll  = gpmix.CLikNormalIso()
#create hyperparm     
hyperparams = gpmix.CGPHyperParams()
hyperparams['covar'] = covar_params
hyperparams['lik'] = lik_params
#cretae GP
gp=gpmix.CGPbase(covar,ll)
#set data
gp.setY(y)
gp.setX(X)
lml = gp.LML(hyperparams)
dlml = gp.LMLgrad(hyperparams)

#optimization
gpopt = gpmix.CGPopt(gp)
gpopt.opt()
opt_params = gp.getParamArray()
lmlo = gp.LML()
t2 = time.time()


print "lml: %.2f -- %.2f" % (lml,lml_)
print "lmlO: %.2f -- %.2f" % (lmlo,lmlo_)
print (t1-t0)
print (t2-t1)
