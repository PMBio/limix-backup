# Copyright(c) 2014, The LIMIX developers (Christoph Lippert, Paolo Francesco Casale, Oliver Stegle)
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.


"""testing script for multi trait covariance stuff"""
import sys
sys.path.append('./..')
sys.path.append('./../../..')

import scipy as SP
import scipy.linalg
import limix as limix
import time
import pdb
import pylab as PL


def scale_k(k, verbose=False):
    c = SP.sum((SP.eye(len(k)) - (1.0 / len(k)) * SP.ones(k.shape)) * SP.array(k))
    scalar = (len(k) - 1) / c
    if verbose:
        print 'Kinship scaled by: %0.4f' % scalar
    k = scalar * k
    return k



SP.random.seed(1)

#1. simulate
#samples:
N = 200
#number of low rank population structure factors
K = 5
#number of SNPs
S = 10000
#number of traits
T=2 

print "Simulating: %d samples, %d factors (genotype), %d SNPs %d traits" % (N,K,S,T)

assert T==2, 'not supported yet!'

#variance components
V00 = 1.0
V11 = 1.0
V01 = 0.9

Vnoise = 1.0

#number of genetic factors
Nsnp_common = 1
Vsnp_common = 0.8
Nsnp_interacting = 1
Vsnp_interacting = 0.8


#1.1 simluate SNPs with low rank structure (population)
Xp = SP.dot(SP.random.randn(N,K),SP.random.randn(K,S))
Xp-= Xp.mean(axis=0)
Xp/= Xp.std(axis=0)
Xr = SP.random.randn(N,S)
X = 0.5*Xp + 0.5*Xr

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


#make all data full scale
Xf = SP.concatenate((X,X),axis=0)
Kpopf = 1.0/Xf.shape[1] * SP.dot(Xf,Xf.T)


#1.6 add fixed effect (SNP)
#TODO: make this proper
Iasso = []
Iinter= []
Yf = SP.zeros_like(Yr)
for ii in xrange(Nsnp_common):
    iis = SP.random.permutation(S)[0]
    w   = SP.sqrt(Vsnp_common)*SP.random.randn()
    ys   = w*X[:,iis]
    #kron [1,1] : in both environments
    Yf   += SP.kron([1,1],ys)[:,SP.newaxis]
    Iasso.append(iis)

for ii in xrange(Nsnp_interacting):
    iis = SP.random.permutation(S)[0]
    w   = SP.sqrt(Vsnp_interacting)*SP.random.randn()
    ys   = w*X[:,iis]
    #kron [1,1] : in both environments
    Yf   += SP.kron([1,0],ys)[:,SP.newaxis]
    Iinter.append(iis)
    
Iasso = SP.array(Iasso)
Iinter = SP.array(Iinter)


#1.7: sum of fixed and random effect
Y = Yf + Yr
#standardize
Y -= Y.mean()
Y/= Y.std()


#2. fitting using limix
GP = {}
#fix covariance, taking population structure
GP['covar_G'] = limix.CFixedCF(Kpopf)
#freeform covariance: requiring number of traits/group (T)
GP['covar_E'] = limix.CCovFreeform(T)

#overall covarianc: product
GP['covar'] = limix.CProductCF()
GP['covar'].addCovariance(GP['covar_G'])
GP['covar'].addCovariance(GP['covar_E'])
#liklihood: gaussian
GP['ll'] = limix.CLikNormalIso()
GP['data'] =  limix.CData()
GP['hyperparams'] = limix.CGPHyperParams()


#Create GP instance
GP['gp']=limix.CGPbase(GP['data'],GP['covar'],GP['ll'])
#set data
GP['gp'].setY(Y)
#input: effectively we require the group for each sample (CCovFreeform requires this)
Xtrain = SP.zeros([Y.shape[0],1])
Xtrain[N::1] = 1        
GP['gp'].setX(Xtrain)

gpopt = limix.CGPopt(GP['gp'])
#constraints: make sure that noise level does not go completel crazy
constrainU = limix.CGPHyperParams()
constrainL = limix.CGPHyperParams()
constrainU['lik'] = +5*SP.ones([1]);
constrainL['lik'] = -5*SP.ones([1]);
gpopt.setOptBoundLower(constrainL);
gpopt.setOptBoundUpper(constrainU);
#filters: we do not need to optimize the overall scaling prameter in front of Kpop



P = []
L = []
#use a number of restarts...
#TODO: think about a more elegant scheme to handle multiple restarts properly
t0=time.time()
for i in xrange(1):
    covar_params =  0.5*SP.random.randn(GP['covar'].getNumberParams())
    lik_params   =  0.5*SP.random.randn(1)
    GP['hyperparams']['covar'] = covar_params
    GP['hyperparams']['lik']   = lik_params
    GP['gp'].setParams(GP['hyperparams'])
    gpopt.opt()
    P.append(GP['gp'].getParamArray())
    L.append(GP['gp'].LML())
im=SP.array(L).argmin()
GP['gp'].setParamArray(P[im])
t1=time.time()

#checking tha gradients work
#print gpopt.gradCheck()
print "needed %.2f seconds for variance component learning" % (t1-t0)

#get optmized variance component
Ke = GP['covar_E'].K()
K = GP['covar'].K()

#variance components are in here:
params_E=GP['covar_E'].getParams()
params_G=GP['covar_G'].getParams()
    
    
K0 = SP.eye(K.shape[0])

#1. Main effect
#covariates
C = SP.ones([2*N,1])
lmm = limix.CLMM()
lmm.setK(scale_k(K))
lmm.setSNPs(Xf)
lmm.setPheno(Y)
#covariates: column of ones
lmm.setCovs(C)
#EmmaX mode with useful default settings
lmm.setEMMAX()
t0 = time.time()
lmm.process()
t1 = time.time()
pv = lmm.getPv().flatten()
print "neede %.2f seconds for GWAS scan" % (t1-t0) 

#comparison with standard Kpop
lmm.setK(Kpopf)
lmm.process()
pvk = lmm.getPv().flatten()


#2. interaction effects (env 0 only)
I = SP.zeros([Y.shape[0],1])
I0 = SP.ones([Y.shape[0],1])

I[0:N]=1
lmi = limix.CInteractLMM()
lmi.setK(scale_k(K))
lmi.setSNPs(Xf)
lmi.setPheno(Y)
lmi.setCovs(SP.concatenate((C,I),axis=1))
lmi.setEMMAX()
lmi.setInter(I)
lmi.setInter0(I0)
lmi.process()
pvI = lmi.getPv().flatten()

#comparison with standard Kpop
lmi.setK(scale_k(Kpopf))
lmi.process()
pvIk = lmi.getPv().flatten()


if 1:
    PL.figure()
    pk=PL.plot(-SP.log10(pvk),'b.',alpha=0.5)    
    p=PL.plot(-SP.log10(pv),'r.',alpha=0.5)
    PL.plot(Iasso,0*SP.ones_like(Iasso),'r*',markersize=15)
    PL.legend([pk,p],['LMM','LMM2trait'],loc='upper left')    

if 1:
    PL.figure()
    pk=PL.plot(-SP.log10(pvIk),'b.',alpha=0.5)    
    p=PL.plot(-SP.log10(pvI),'r.',alpha=0.5)
    PL.plot(Iinter,0*SP.ones_like(Iinter),'r*',markersize=15)
    PL.legend([pk,p],['LMM','LMM2trait'],loc='upper left')        
        
