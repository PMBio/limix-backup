"""testing script for multi trait covariance stuff"""
import sys
sys.path.append('./..')
sys.path.append('./../../..')

import scipy as SP
import scipy.linalg
import gpmix as mmtk
import time
import pdb
import pylab as PL


#1. simulate
#samples:
N = 200
#number of low rank population structure factors
K = 5
#number of SNPs
S = 100000
#number of traits
T=2 

print "Simulating: %d samples, %d factors (genotype), %d SNPs %d traits" % (N,K,S,T)

assert T==2, 'not supported yet!'

#variance components
V00 = 1.0
V11 = 1.0
V01 = 0.001
Vnoise = 1.0

#number of genetic factors
Nsnp_common = 1
Vsnp_common = 0.8
Nsnp_interacting = 0
Vsnp_interacting = 0.8


#1.1 simluate SNPs with low rank structure (population)
X = SP.dot(SP.random.randn(N,K),SP.random.randn(K,S))
X-= X.mean(axis=0)
X/= X.std(axis=0)

#1.2 population structure covariance
Kpop = 1.0/X.shape[1] *SP.dot(X,X.T)

#1.3 simulate covariance structure between traits without SNP effect
C = SP.zeros([2,2])
C[0,0] = V00
C[1,1] = V11
C[0,1] = V01
C[1,0] = V01
#1.4 create overall kernel as kronecker product
K = SP.kron(C,Kpop) 
K +=  Vnoise * SP.eye(K.shape[0])

#1.5 sample from kernel
L = SP.linalg.cholesky(K).T
Yr = SP.dot(L,SP.random.randn(K.shape[0],1))  

#1.6 add fixed effect (SNP)
#TODO: make this proper
Yf = SP.zeros_like(Yr)
for ii in xrange(Nsnp_common):
    iis = SP.random.permutation(S)[0]
    w   = SP.sqrt(Vsnp_common)*SP.random.randn()
    ys   = w*X[:,ii]
    #kron [1,1] : in both environments
    Yf   += SP.kron([1,1],ys)[:,SP.newaxis]

for ii in xrange(Nsnp_interacting):
    iis = SP.random.permutation(S)[0]
    w   = SP.sqrt(Vsnp_interacting)*SP.random.randn()
    ys   = w*X[:,ii]
    #kron [1,1] : in both environments
    Yf   += SP.kron([1,1],ys)[:,SP.newaxis]


#1.7: sum of fixed and random effect
Y = Yf + Yr
#standardize
Y -= Y.mean()
Y/= Y.std()


#make all data full scale
Xf = SP.concatenate((X,X),axis=0)
Kpopf = 1.0/Xf.shape[1] * SP.dot(Xf,Xf.T)

#2. fitting using mmtk
GP = {}
#fix covariance, taking population structure
GP['covar_G'] = mmtk.CFixedCF(Kpopf)
#freeform covariance: requiring number of traits/group (T)
GP['covar_E'] = mmtk.CCovFreeform(T)

#overall covarianc: product
GP['covar'] = mmtk.CProductCF()
GP['covar'].addCovariance(GP['covar_G'])
GP['covar'].addCovariance(GP['covar_E'])
#liklihood: gaussian
GP['ll'] = mmtk.CLikNormalIso()
GP['data'] =  mmtk.CData()
GP['hyperparams'] = mmtk.CGPHyperParams()
covar_params = SP.zeros([GP['covar'].getNumberParams()])
lik_params   =  SP.log([0.1])
GP['hyperparams']['covar'] = covar_params
GP['hyperparams']['lik']   = lik_params
#Create GP instance
GP['gp']=mmtk.CGPbase(GP['data'],GP['covar'],GP['ll'])
GP['gp'].setParams(GP['hyperparams'])
    
#set data
GP['gp'].setY(Y)

#input: effectively we require the group for each sample (CCovFreeform requires this)
Xtrain = SP.zeros([Y.shape[0],1])
Xtrain[N::1] = 1        
GP['gp'].setX(Xtrain)
#constraints
constrainU = mmtk.CGPHyperParams()
constrainL = mmtk.CGPHyperParams()
constrainU['lik'] = +5*SP.ones_like(lik_params);
constrainL['lik'] = -5*SP.ones_like(lik_params);

gpopt = mmtk.CGPopt(GP['gp'])
#checking tha gradients work
#print gpopt.gradCheck()
gpopt.setOptBoundLower(constrainL);
gpopt.setOptBoundUpper(constrainU);
t0=time.time()
gpopt.opt()
t1=time.time()
print "needed %.2f seconds for variance component learning" % (t1-t0)

#get optmized variance component
Ke = GP['covar_E'].K()
K = GP['covar'].K()

#variance components are in here:
params_E=GP['covar_E'].getParams()
params_G=GP['covar_G'].getParams()
    
#get full covariance and use for testijng
lmm = mmtk.CLMM()
lmm.setK(K)
lmm.setSNPs(Xf)
lmm.setPheno(Y)
#covariates: column of ones
lmm.setCovs(SP.ones([2*N,1]))
#EmmaX mode with useful default settings
lmm.setEMMAX()
t0 = time.time()
lmm.process()
t1 = time.time()

print "neede %.2f seconds for GWAS scan" % (t1-t0) 
pv = lmm.getPv().flatten()

PL.figure()
PL.plot(-SP.log10(pv))    
        
