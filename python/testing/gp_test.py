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

covar  = gpmix.CCovSqexpARD(n_dimensions)
covar_ = se.SqexpCFARD(n_dimensions)

ll  = gpmix.CLikNormalIso()
ll_ = lik.GaussLikISO()

params = SP.random.randn(n_dimensions+1)
hyperparams = {'covar':params,'lik':SP.log([1])}     

gp_ = GP.GP(covar_,likelihood=ll_,x=X,y=y)

#GP
covar.setParams(hyperparams['covar'])
ll.setParams(hyperparams['lik'])
gp=gpmix.CGPbase(covar,ll)
covar.setX(X)
ll.setX(X)
gp.setY(y)

lml = gp.LML()
lml_ = gp_.LML(hyperparams)
#
