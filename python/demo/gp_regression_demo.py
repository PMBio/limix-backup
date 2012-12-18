import sys
sys.path.append('./../../build_mac/src/interfaces/python')

import limix
import scipy as SP
import pdb

#set random seed
SP.random.seed(1)
#genrate toy data (x,y)
#dimensions : number of features
n_dimensions=1
n_samples = 30
X = SP.randn(n_samples,n_dimensions)
y = SP.dot(X,SP.randn(n_dimensions,1))
y += 0.2*SP.randn(y.shape[0],y.shape[1])


#array of X values where we would like to evaluate the prediction
#test set
Xs = SP.linspace(X.min()-3,X.max()+3)[:,SP.newaxis]

#initialize LIMX objet
#1. determine covariance function:
# Squared expontential, Gaussian kernel 
covar  = limix.CCovSqexpARD(n_dimensions)
#2. likelihood: Gaussian noise
ll  = limix.CLikNormalIso()

#startin parmaeters are set random
covar_params = SP.random.randn(covar.getNumberParams())
lik_params = SP.random.randn(ll.getNumberParams())

#create hyperparameter object
hyperparams0 = limix.CGPHyperParams()
#set fields "covar" and "lik"
hyperparams0['covar'] = covar_params
hyperparams0['lik'] = lik_params

#cretae GP object
gp=limix.CGPbase(covar,ll)
#set data
#outputs
gp.setY(y)
#inputs
gp.setX(X)
#startgin parameters
gp.setParams(hyperparams0)

#starting marginal liklelihood and derivative
lml0 = gp.LML()
dlml0 = gp.LMLgrad()

#set optimization constriats (optional)
constrainU = limix.CGPHyperParams()
constrainL = limix.CGPHyperParams()
constrainU['covar'] = +10*SP.ones_like(covar_params);
constrainL['covar'] = -10*SP.ones_like(covar_params);
constrainU['lik'] = +5*SP.ones_like(lik_params);
constrainL['lik'] = -5*SP.ones_like(lik_params);

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
print SP.exp(2*hyperparams0['covar'])
print SP.exp(2*hyperparams0['lik'])

print "optimized hyperparams"
#print hyperparams_opt
print SP.exp(2*hyperparams_opt['covar'])
print SP.exp(2*hyperparams_opt['lik'])

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
