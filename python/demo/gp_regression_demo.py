# Copyright(c) 2014, The LIMIX developers (Christoph Lippert, Paolo Francesco Casale, Oliver Stegle)
# All rights reserved.
#
# LIMIX is provided under a 2-clause BSD license.
# See license.txt for the complete license.

import sys
sys.path.append('./../../build_mac/src/interfaces/python')

import limix
import scipy as SP
import pdb

#set random seed
SP.random.seed()
#genrate toy data (x,y)
#dimensions : number of features
n_dimensions=1
n_samples = 20
X = SP.rand(n_samples,n_dimensions)

#array of X values where we would like to evaluate the prediction
#test set
Xs = SP.linspace(X.min(),X.max())[:,SP.newaxis]

pdb.set_trace()

#initialize LIMX objet
#1. determine covariance function:
# Squared expontential, Gaussian kernel 
covar  = limix.CCovSqexpARD(n_dimensions)
#2. likelihood: Gaussian noise
ll  = limix.CLikNormalIso()

#startin parmaeters are set random
covar_params = SP.array([1,0.2])
lik_params = SP.array([0.01])

#create hyperparameter object
hyperparams0 = limix.CGPHyperParams()
#set fields "covar" and "lik"
hyperparams0['covar'] = covar_params
hyperparams0['lik'] = lik_params

#cretae GP object
gp=limix.CGPbase(covar,ll)
#inputs
gp.setX(X)
#startgin parameters
gp.setParams(hyperparams0)
#generate and set outputs
y=SP.random.multivariate_normal(SP.zeros(X.shape[0]),(gp.getCovar()).K()+(gp.getLik()).K())
gp.setY(y)

#startin parmaeters are set random
covar_params = SP.exp(SP.random.randn(covar.getNumberParams()))
lik_params = SP.exp(SP.random.randn(ll.getNumberParams()))

#starting marginal liklelihood and derivative
lml0 = gp.LML()
dlml0 = gp.LMLgrad()

#set optimization constriats (optional)
constrainU = limix.CGPHyperParams()
constrainL = limix.CGPHyperParams()
constrainU['covar'] = float('inf')*SP.ones_like(covar_params);
constrainL['covar'] = SP.zeros_like(covar_params);
constrainU['lik'] = float('inf')*SP.ones_like(lik_params);
constrainL['lik'] = SP.zeros_like(lik_params);

#create optimization object
gpopt = limix.CGPopt(gp)
#set constrants
gpopt.setOptBoundLower(constrainL);
gpopt.setOptBoundUpper(constrainU);
#run
gpopt.opt()

#get optimal parameters and LML
hyperparams_opt = gp.getParams()
lml_opt = gp.LML()

#prediction
Xmean = gp.predictMean(Xs)
Xstd = gp.predictVar(Xs)

#print out stuff

print "initial hyperparams"
#print hyperparams0
print "initial marginal likelihood"
print -lml0
print hyperparams0['covar']
print hyperparams0['lik']

print "optimized hyperparams"
#print hyperparams_opt
print hyperparams_opt['covar']
print hyperparams_opt['lik']

print "error estimated through laplace approximation"
stdL=gp.getStd_laplace()
print stdL['covar']
print stdL['lik']

print "optimized marginal likelihood"
print -lml_opt
print "gradients of optimization at optimum"
print gp.LMLgrad()



#plottin
import pylab as PL
PL.ion()
PL.figure()
PL.plot(X,y,'b.')
PL.plot(Xs,Xmean)
PL.plot(Xs,Xmean+SP.sqrt(Xstd))
PL.plot(Xs,Xmean-SP.sqrt(Xstd))
PL.show()

pdb.set_trace()
