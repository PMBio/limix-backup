import sys
sys.path.append('./..')
sys.path.append('./../../../pygp')


import limix
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

SP.random.seed(1)

n_dimensions=1
n_samples = 30
X = SP.randn(n_samples,n_dimensions)
y = SP.dot(X,SP.randn(n_dimensions,1))
y += 0.2*SP.randn(y.shape[0],y.shape[1])

covar_params = SP.random.randn(n_dimensions+1)
lik_params = SP.random.randn(1)

Xs = SP.linspace(X.min()-3,X.max()+3)[:,SP.newaxis]

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

pdb.set_trace()


#GPMIX:
covar  = limix.CCovSqexpARD(n_dimensions)
ll  = limix.CLikNormalIso()
data = limix.CData()

#create hyperparm     
hyperparams = limix.CGPHyperParams()
hyperparams['covar'] = covar_params
hyperparams['lik'] = lik_params

#cretae GP
gp=limix.CGPbase(covar,ll,data)
#set data
gp.setY(y)
gp.setX(X)
gp.setParams(hyperparams)
lml = gp.LML()
dlml = gp.LMLgrad()

#build constraints
constrainU = limix.CGPHyperParams()
constrainL = limix.CGPHyperParams()
constrainU['covar'] = +10*SP.ones_like(covar_params);
constrainL['covar'] = -10*SP.ones_like(covar_params);
constrainU['lik'] = +5*SP.ones_like(lik_params);
constrainL['lik'] = -5*SP.ones_like(lik_params);


mask = limix.CGPHyperParams()
covar_mask = SP.ones(covar_params.shape[0])
covar_mask[0] = 1
covar_mask[1] = 1

mask['covar'] = covar_mask
#mask['lik'] = SP.ones(lik_params.shape[0])


gpopt = limix.CGPopt(gp)
gpopt.setOptBoundLower(constrainL);
gpopt.setOptBoundUpper(constrainU);
gpopt.setParamMask(mask)
gpopt.opt()
opt_params = gp.getParamArray()
lmlo = gp.LML()

#prediction
Xmean = gp.predictMean(Xs)
Xstd = gp.predictVar(Xs)

#predict
[M,S] = gp_.predict(opt_params_,Xs)


import pylab as PL
PL.figure()
PL.plot(X,y,'b.')
PL.plot(Xs,M)
PL.plot(Xs,M+SP.sqrt(S))
PL.plot(Xs,M-SP.sqrt(S))


PL.figure()
PL.plot(X,y,'b.')
PL.plot(Xs,Xmean)
PL.plot(Xs,Xmean+SP.sqrt(Xstd))
PL.plot(Xs,Xmean-SP.sqrt(Xstd))
