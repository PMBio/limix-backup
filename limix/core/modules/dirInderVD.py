import h5py
import scipy as sp
from limix.core.gp.gp_base import GP
from limix.core.mean.mean_base import mean_base as lin_mean
from limix.core.covar.dirIndirCov import DirIndirCov
from limix.core.covar.fixed import FixedCov 
from limix.core.covar.combinators import SumCov
from limix.core.utils.normalization import covar_rescaling_factor
import ipdb

class DirIndirVD():

    def __init__(self, pheno = None, kinship = None, cage = None, covs = None):

        assert pheno is not None, 'Specify pheno!'
        assert kinship is not None, 'Specify kinship!'
        assert cage is not None, 'Specify cage!'

        if len(cage.shape)==1:
            cage = cage[:,sp.newaxis]

        self.N = pheno.shape[0]

        if covs is None:
            covs = sp.ones((self.N,1)) 

        # build design matrix and cage covariates
        uCage = sp.unique(cage)
        W = sp.zeros((self.N,uCage.shape[0])) # cage covariates n_samples x n_cages
        for cv_i, cv in enumerate(uCage):
            W[:,cv_i] = 1.*(cage[:,0]==cv)
        WW = sp.dot(W,W.T)
        Z  = WW - sp.eye(self.N)

        # define mean
        self.mean = lin_mean(pheno,covs)

        # define covariance matrices
        self._genoCov = DirIndirCov(kinship,Z)
        self._envCov = DirIndirCov(sp.eye(self.N),Z)
        self._noisCov = FixedCov(WW)
        covar = SumCov(self._genoCov,self._envCov,self._noisCov)

        # define gp
        self._gp = GP(covar=covar,mean=self.mean)

    def optimize(self, calc_ste = False, verbose = True):
        if 0:
            # trial for inizialization it is complicated though
            cov = sp.array([[0.2,1e-4],[1e-4,1e-4]])
            self._genoCov.setCovariance(cov)
            self._envCov.setCovariance(cov)
            self._noisCov.scale = 0.2
        else:
            self._gp.covar.setRandomParams()

        # optimization
        conv, info = self._gp.optimize(calc_ste = calc_ste)

        # return stuff
        R = {}
        R['conv'] = conv
        R['grad'] = info['grad']
        R['LML']  = self._gp.LML() 

        # panda dataframe here?
        R['var_Ad'] = self._genoCov.covff.K()[0,0]
        R['var_As'] = self._genoCov.covff.K()[1,1]
        R['sigma_Ads'] = self._genoCov.covff.K()[0,1]
        R['var_Ed'] = self._envCov.covff.K()[0,0]
        R['var_Es'] = self._envCov.covff.K()[1,1]
        R['sigma_Eds'] = self._envCov.covff.K()[0,1]
        R['b'] = self.mean.b
        R['var_C'] = self._noisCov.scale

        # TODO: calculate rho

        if verbose:
            print '\ngeno variance components'
            for key in ['var_Ad','var_As','sigma_Ads']:
                print '%s:'%key, R[key]
            print '\nenv variance components'
            for key in ['var_Ed','var_Es','sigma_Eds']:
                print '%s:'%key, R[key]
            print '\ncage variance components'
            for key in ['var_C']:
                print '%s:'%key, R[key]
            print ''

        return R

    def getResidual(self):
        K = self._envCov.K() + self._noisCov.K()
        var = covar_rescaling_factor(K)
        K /= var
        R = {'normalized_residual': K,
             'variance_explained': var}
        return R

    def getDirIndirGenoCovar(self):
        return self._genoCov.dirIndirCov_K()

    def getDirIndirEnvCovar(self):
        return self._envCov.dirIndirCov_K()

    def getDirInderNoisVar(self):
        return self._noisCov.scale

if __name__=='__main__':

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

    if 1:
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
        rv = vc.optimize()
        print 'lml:', rv['LML']
        res = vc.getResidual(calc_ste = True)
        print 'residual:', res['variance_explained']
        ipdb.set_trace()
        vc._genoCov.variance
        vc._genoCov.variance_ste
        vc._genoCov.correlation
        vc._genoCov.correlation_ste

