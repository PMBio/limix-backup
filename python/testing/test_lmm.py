import sys

import os
os.chdir('/kyb/agbs/stegle/work/projects/GPmix/python/testing')

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
    y_ = Y[:,ip:ip+1]
    Iok = (~SP.isnan(y_)).all(axis=1)
    y_ = y_[Iok]
    X_ = X[Iok,::1]
    K = 1./X_.shape[1]*SP.dot(X_,X_.T)
    C_ = SP.ones([X_.shape[0],1])
        
    if 1:
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
    #likelihood ratios
    t3 = time.time()
    lm.process()    
    t4 = time.time()
    pv1_llr = lm.getPv()
    if 1:
        #ftests
        lm.setTestStatistics(gpmix.CLMM.TEST_F)
        lm.process()    
        pv1_ft = lm.getPv()
    
    if 0:
        Nperm = 10
        PVp = SP.zeros([Nperm,X_.shape[1]])
        t5=time.time()
        for i in xrange(Nperm):
            perm = SP.random.permutation(y_.shape[0])
            lm.setPermutation(perm)
            lm.process()    
            PVp[i,:] = lm.getPv().squeeze()
        t6=time.time()
               
    
    print SP.absolute(pv1_llr-pv0).max()
    if 0:
        PL.plot(-SP.log(pv1[0,:]))
    
