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

    else:
        # import data
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

    # define model and optimize
    vc = DirIndirVD(pheno=Y, kinship=kinship, cage=cage, covs = covs)
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

