"""testing script for multi trait covariance stuff"""

import sys
#1. add path to mixedtk
sys.path.append('./..')

import scipy as SP
import scipy.linalg
import limix as mmtk
import time
import pdb
import pylab as PL
import modules.multiTraitQTL as MM


if __name__ == '__main__':
    
    #1. simulate something trivial
    
    SP.random.seed(1)
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
    
    #variance components: genotype
    Vg00 = 1.0
    Vg11 = 1.0
    Vg01 = 0.8
    
    #variance components: env
    Ve00 = 0.2
    Ve11 = 0.2
    Ve01 = 0.0
    
    #number of genetic factors
    Nsnp_common = 1
    Vsnp_common = 0.8
    #interactions (env. specific effects)
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
    Kpop = MM.scale_k(Kpop)
    
    #1.3 simulate covariance structure between traits without SNP effect
    Cg = SP.zeros([2,2])
    Cg[0,0] = Vg00
    Cg[1,1] = Vg11
    Cg[0,1] = Vg01
    Cg[1,0] = Vg01
    
    #1.4 genotype component:
    Kg = SP.kron(Cg,Kpop)
    
    #1.5 env. component
    Ce = SP.zeros([2,2])
    Ce[0,0] = Ve00
    Ce[1,1] = Ve11
    Ce[0,1] = Ve01
    Ce[1,0] = Ve01
    Ke = SP.kron(Ce,SP.eye(N))
     
    #Kronecker sum covariance
    K = Kg + Ke
    
    #1.5 sample from kernel
    L = SP.linalg.cholesky(K).T
    Yr = SP.dot(L,SP.random.randn(K.shape[0],1))  
    
    #make all data full scale
    Xf = SP.concatenate((X,X),axis=0)
    Kpopf = 1.0/Xf.shape[1] * SP.dot(Xf,Xf.T)
    Kpopf = MM.scale_k(Kpopf)
        
    
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
    if 0:
        Y -= Y.mean()
        Y/= Y.std()
    
    #env. indicator variable (0/1)
    E = SP.zeros([Y.shape[0],1])
    E[N::1] = 1  
    
    #genotye identity        
    Kgeno = SP.kron(SP.ones([2,2]),SP.eye(N))
    
    #####ANALYSIS#####
    print "Variance component fitting"
    
    tt = MM.CMMT(X=Xf,E=E,Y=Y.copy(),Kpop=Kpopf,Kgeno=Kgeno,T=T)
    t0 =time.time()
    tt.fitVariance()
    t1 = time.time()
    print "--done (%.2f seconds)" % (t1-t0)
    
    VE = tt.VE
    VG = tt.VG
    
        
    print "fitted varaince components"
    print "VE"
    print VE
    print "VG"
    print VG
    
    #GWA scan
    print "Main effect GWAS scans (2)"
    t0= time.time()
    pv_Kpop = tt.GWAmain(useK='Kpop')
    pv_multiTrait = tt.GWAmain(useK='multi_trait')
    t1 = time.time()
    print "--done (%.2f seconds)" %(t1-t0)
    
    print "Interaction effect GWAS scans (2)"
    t0=time.time()
    I = SP.zeros([2*N,1])
    I[0:N] = 1
    I0 = SP.ones([2*N,1])
    pvi_Kpop=tt.GWAinter(useK='Kpop',I=I,I0=I0)
    pvi_multiTrait=tt.GWAinter(useK='multi_trait',I=I,I0=I0)
    t1=time.time()
    print "--done (%.2f seconds)" %(t1-t0)
    
    
    if 1:
        PL.figure()
        PL.subplot(211)
        PL.title('Kpop')
        PL.plot(-SP.log(pv_Kpop),'b.')
        PL.plot(Iasso,SP.zeros(len(Iasso)),'r*',markersize=15)
        PL.subplot(212)
        PL.title('Kmultitrait')
        PL.plot(-SP.log(pv_multiTrait),'b.')
        PL.plot(Iasso,SP.zeros(len(Iasso)),'r*',markersize=15)
        
    
        PL.figure()
        PL.subplot(211)
        PL.title('Kpop')
        PL.plot(-SP.log(pvi_Kpop),'b.')
        PL.plot(Iinter,SP.zeros(len(Iinter)),'r*',markersize=15)
        
        PL.subplot(212)
        PL.plot(-SP.log(pvi_multiTrait),'b.')
        PL.plot(Iinter,SP.zeros(len(Iinter)),'r*',markersize=15)
        PL.title('Kmultitrait')
        
        