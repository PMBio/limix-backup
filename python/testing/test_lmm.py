import sys

import os
#os.chdir('/kyb/agbs/stegle/work/projects/GPmix/python/testing')

sys.path.append('./../../pygp')
sys.path.append('./..')
sys.path.append('./../../..')


import h5py
import scipy as SP
import gpmix
import panama.core.lmm.lmm as lmm
import pylab as PL
import pdb
import time


if __name__ == '__main__':
    
    
    
    if 0:
        K = SP.eye(100)
        X = SP.random.randn(100,1000)
        C = SP.ones([100,1])
        I = 1.0*(SP.random.rand(100,2)<0.2)
        I0 = 1.0*SP.ones([100,1])
        
        y = 0.2*(I[:,0:1]*X[:,333:333+1]) 
        y/=y.std()
        y+= 0.2*SP.random.randn(y.shape[0],y.shape[1])
        lm = gpmix.CInteractLMM()
        lm.setTestStatistics(gpmix.CLMM.TEST_F)
        lm.setK(K)
        lm.setSNPs(X)
        lm.setPheno(y)
        lm.setCovs(SP.concatenate((C,I[:,0:1]),axis=1))
        lm.setInter(I[:,0:1])
        lm.setInter0(I0)

        pdb.set_trace()
        lm.process()
        pv1_llr = lm.getPv()
            
        PL.plot(-SP.log(pv1_llr).ravel())
        pdb.set_trace()
        pass
    
    
    hd = h5py.File('/kyb/agbs/stegle/work/projects/warpedlmm/data/Nordborg_data.h5py','r')
    geno = hd['geno']
    pheno = hd['pheno']

    phenotype_names = hd['pheno/phenotype_names'][:]
    Npheno = phenotype_names.shape[0]
    #2. geno/pheno
    
    geno_index = pheno['geno_index'][:]
    #resort in increasing order
    Is = geno_index.argsort()
    geno_index = geno_index[Is]

    Y = pheno['Y'][:][Is]
    X = SP.asarray(geno['x'][:,geno_index][:].T,dtype='float')
    #center genotype
    X-=X.mean()
    X/-X.std()

    ip = 7
    y_ = Y[:,ip:ip+5]
    Iok = (~SP.isnan(y_)).all(axis=1)
    y_ = y_[Iok]
    X_ = X[Iok,::1]
    K = 1./X_.shape[1]*SP.dot(X_,X_.T)
    C_ = SP.ones([X_.shape[0],1])
        
    if 1:
        #OLD
        #population covariance
        t0 = time.time()
        [lod,pv0] = lmm.train_associations(X_,y_,K,C_)
        t1 = time.time()
  
    #gpmix
    lm = gpmix.CLMM()
    lm.setK(K)
    lm.setSNPs(X_)
    lm.setPheno(y_)
    lm.setCovs(C_)
    lm.setEMMAX()    
    #condition on SNP
    #C_ = SP.concatenate((C_,X_[:,89790:89790+1]),axis=1)
    #lm.setCovs(C_)
    
    
    #likelihood ratios
    if 1:
        t3 = time.time()
        lm.setTestStatistics(gpmix.CLMM.TEST_LLR)
        lm.process()    
        t4 = time.time()
        pv1_llr = lm.getPv()
        nll0 = lm.getNLL0()
        nllalt = lm.getNLLAlt()
    if 1:
        #ftests
        t5= time.time()
        lm.setTestStatistics(gpmix.CLMM.TEST_F)
        lm.process()    
        pv1_ft = lm.getPv()
        print lm.getNLL0()
        print lm.getNLLAlt()       
        t6= time.time()
        
    if 0:
        import pylab as PL
        PL.plot(-SP.log(pv1_llr.ravel()),'b.')
        PL.plot(-SP.log(pv1_ft.ravel()),'r.')
        
    
    
    
    
