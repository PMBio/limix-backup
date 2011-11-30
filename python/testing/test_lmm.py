import sys
sys.path.append('./../build/src/python_interface')
sys.path.append('./../pygp')
sys.path.append('./../../')


import h5py
import scipy as SP
import gpmix
import panama.core.lmm.lmm as lmm


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
    y_ = Y[:,ip]
    Iok = ~SP.isnan(y_)
    y_ = y_[Iok]
    X_ = X[Iok,::10]

    y_ = y_[:,SP.newaxis]
    
    #population covariance
    K = 1./X_.shape[1]*SP.dot(X_,X_.T)
    C_ = SP.ones([X_.shape[0],1])
    [lod,pv0] = lmm.train_associations(X_,y_,K,C_)

    #gpmix
    lm = gpmix.CLmm()
    lm.setK(K)
    lm.setSNPs(X_)
    lm.setPheno(y_)
    lm.setCovs(C_)
    lm.process()
    pv1 = lm.getPv()
    
    
