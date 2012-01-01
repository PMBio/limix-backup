import sys
sys.path.append('./..')
sys.path.append('./../../../pygp')


import gpmix
import pygp.covar.linear as lin
import pygp.likelihood as lik
import pygp.gp.gp_base as GP
import pygp.covar.se as se
import pygp.covar.gradcheck as GC
import pygp.covar.combinators as comb
import scipy as SP
import pdb



n_dimensions=3
X = SP.randn(10,n_dimensions)
y = SP.randn(10,1)

covar_params = SP.random.randn(n_dimensions+1)
lik_params = SP.random.randn(1)

#pygp:
covar_ = se.SqexpCFARD(n_dimensions)
ll_ = lik.GaussLikISO()
hyperparams_ = {'covar':covar_params,'lik':lik_params}
gp_ = GP.GP(covar_,likelihood=ll_,x=X,y=y)
lml_ = gp_.LML(hyperparams_)
dlml_ = gp_.LMLgrad(hyperparams_)

#gpmix
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

print "lml: %.2f -- %.2f" % (lml,lml_)
