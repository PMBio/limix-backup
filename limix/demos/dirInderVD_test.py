import h5py
import scipy as sp
import ipdb
from limix.modules.dirIndirVD import DirIndirVD

if __name__=='__main__':

    if 0:

        # generate data
        n = 100
        f = 2
        X  = 1.*(sp.rand(n,f)<0.2)
        X -= X.mean(0); X /= X.std(0)
        kinship  = sp.dot(X,X.T)
        kinship /= kinship.diagonal().mean()
        cage = sp.zeros((n,1))
        for i in range(n/2):
            cage[2*i:2*i+2] = i
        Y = sp.randn(n,1)
        covs = None

    elif 0:
        # import data considers only cagemates that are both genotypes and phenotyped
        in_file = '/Users/casale/Desktop/rat/dirIndirVD/data/HSrats_noHaplotypes.hdf5'
        f = h5py.File(in_file,'r')

        # get sample ID
        geno_sampleID = f['kinships']['genotypes_IBS']['cols_subjects']['outbred'][:]
        sampleID = f['phenotypesNcovariates']['rows_subjects']['outbred'][:]
        has_geno = sp.array([sampleID[i] in geno_sampleID for i in range(sampleID.shape[0])])

        # read trait and covariantes 
        trait = 'Distance0_30_bc'
        measures = f['phenotypesNcovariates']['cols_measures']['measures'][:]
        Ip = measures==trait
        covs = f['phenotypesNcovariates']['cols_measures']['covariates2use'][Ip][0].split(',')
        Ic = sp.zeros(Ip.shape[0],dtype=bool)
        for cov in covs:    Ic = sp.logical_or(Ic,measures==cov)
        Y = f['phenotypesNcovariates']['array'][Ip,:].T
        covs = f['phenotypesNcovariates']['array'][Ic,:].T
        Is = sp.logical_and((covs!=-999).all(1),Y[:,0]!=-999)
        Is = sp.logical_and(has_geno,Is)
        Y = Y[Is,:]; covs = covs[Is,:]
        cage = f['phenotypesNcovariates']['rows_subjects']['cage'][Is]
        sampleID = sampleID[Is]

        # normalize pheno (not needed if not for numerical stability)
        Y-=Y.mean(0)
        Y/=Y.std(0)

        # grab kinship
        idxs = sp.array([sp.where(geno_sampleID==sampleID[i])[0][0] for i in range(sampleID.shape[0])])
        kinship = f['kinships']['genotypes_IBS']['array'][:][idxs][:,idxs]

    else:

        # import data - considers all cagemates that are genotyped (do not need to be phenotyped)
        in_file = '/Users/casale/Desktop/rat/dirIndirVD/data/HSrats_noHaplotypes.hdf5'
        f = h5py.File(in_file,'r')

        trait = 'Distance0_30_bc'

        #0. load full data that will be used for both focal and cagemates
        cage_full = f['phenotypesNcovariates']['rows_subjects']['cage'][:]
        cage_full_id = f['phenotypesNcovariates']['rows_subjects']['outbred'][:]
        K_full = f['kinships']['genotypes_IBS']['array'][:]
        K_full_id = f['kinships']['genotypes_IBS']['cols_subjects']['outbred'][:]

        #1. focal animals
        sampleID = f['phenotypesNcovariates']['rows_subjects']['outbred'][:]
        has_geno = sp.array([sampleID[i] in K_full_id for i in range(sampleID.shape[0])])
        measures = f['phenotypesNcovariates']['cols_measures']['measures'][:]
        Ip = measures==trait
        covs = f['phenotypesNcovariates']['cols_measures']['covariates2use'][Ip][0].split(',')
        Ic = sp.zeros(Ip.shape[0],dtype=bool)
        for cov in covs:    Ic = sp.logical_or(Ic,measures==cov)
        Y = f['phenotypesNcovariates']['array'][Ip,:].T  # Y is always focal
        covs = f['phenotypesNcovariates']['array'][Ic,:].T  # covs is always focal
        Is = sp.logical_and((covs!=-999).all(1),Y[:,0]!=-999)
        Is = sp.logical_and(has_geno,Is)
        Is = sp.logical_and(cage_full!='NA',Is) # this condition defines focal animals
        Y = Y[Is,:]; covs = covs[Is,:]; cage = cage_full[Is]
        sampleID = sampleID[Is] #this are the samples of focal animal
        idxs = sp.array([sp.where(K_full_id==sampleID[i])[0][0] for i in range(sampleID.shape[0])])
        K = K_full[idxs,:][:,idxs] #this is K focal

        #2. cagemates 
        Imatch = sp.nonzero(cage_full_id[:,sp.newaxis]==K_full_id)
        cage_cm = cage_full[Imatch[0]]
        K_cm = K_full[Imatch[1],:][:,Imatch[1]]
        sampleID_cm = cage_full_id[Imatch[0]]
        has_cage = cage_cm != 'NA'
        cage_cm = cage_cm[has_cage]
        sampleID_cm = sampleID_cm[has_cage]
        K_cm = K_cm[has_cage,:][:,has_cage]

        #3. focal x cagemate cross covariance
        idx_1 = sp.array([sp.where(K_full_id==sampleID[i])[0][0] for i in range(sampleID.shape[0])])
        idx_2 = sp.array([sp.where(K_full_id==sampleID_cm[i])[0][0] for i in range(sampleID_cm.shape[0])])
        K_cross = K_full[idx_1,:][:,idx_2]

    Y -= Y.mean(0)
    Y /= Y.std(0)

    data = {'pheno': Y,
            'kinship': K,
            'cage': cage,
            'covs': covs,
            'sampleID': sampleID,
            'kinship_cm': K_cm,
            'kinship_cross': K_cross,
            'cage_cm': cage_cm,
            'sampleID_cm': sampleID_cm}

    ipdb.set_trace()

    # define model and optimize
    vc = DirIndirVD(**data)
    for i in range(10):
        rv = vc.optimize(calc_ste = True)
        print 'lml:', rv['LML']
        res = vc.getResidual()
        print 'residual:', res['variance_explained']
        print 'geno cov'
        print vc._genoCov.dirIndirCov_K()
        print 'ste on geno cov'
        print vc._genoCov.dirIndirCov_K_ste()
        ipdb.set_trace()

