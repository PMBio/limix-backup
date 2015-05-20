import h5py
import scipy as sp
from limix.core.gp.gp_base import GP
from limix.core.mean.mean_base import mean_base as lin_mean
from limix.core.covar.dirIndirCov import DirIndirCov
from limix.core.covar.fixed import FixedCov 
from limix.core.covar.combinators import SumCov
import ipdb

class DirIndirVD():

    def __init__(self, pheno = None, kinship = None, cage = None, covs=None):

        assert pheno is not None, 'Specify pheno!'
        assert kinship is not None, 'Specify kinship!'
        assert cage is not None, 'Specify cage!'

        if covs is None:
            covs = sp.ones((n,1)) 

        self.N = pheno.shape[0]

        # build design matrix and cage covariates
        uCage = sp.unique(cage)
        W = sp.zeros((self.N,uCage.shape[0])) # cage covariates n_samples x n_cages
        for cv_i, cv in enumerate(uCage):
            W[:,cv_i] = 1.*(cage[:,0]==cv)
        WW = sp.dot(W,W.T)
        Z  = WW - sp.eye(self.N)

        # define mean
        mean = lin_mean(pheno,covs)

        # define covariance matrices
        self._genoCov = DirIndirCov(kinship,Z)
        self._envCov = DirIndirCov(sp.eye(self.N),Z)
        self._noisCov = FixedCov(WW)
        covar = SumCov(self._genoCov,self._envCov,self._noisCov)

        # define gp
        self._gp = GP(covar=covar,mean=mean)

    def optimize(self):
        if 0:
            # trial for inizialization it is complicated though
            cov = sp.array([[0.2,1e-4],[1e-4,1e-4]])
            self._genoCov.setCovariance(cov)
            self._envCov.setCovariance(cov)
            self._noisCov.scale = 0.2
        else:
            self._gp.covar.setRandomParams()

        # optimization
        conv, info = self._gp.optimize()

        ipdb.set_trace()
        # return stuff
        R = {}
        R['conv'] = conv
        R['grad'] = info['grad']
        R['LML']  = self._gp.LML() 

        # panda dataframe here?
        R['var_Ad'] = 1
        R['var_As'] = 1
        R['rho_Ads'] = 1
        R['var_Ed'] = 1
        R['var_Es'] = 1
        R['rho_Eds'] = 1
        R['var_C'] = 1

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

    if 1:
        # import data
        in_file = '/Users/casale/Desktop/rat/dirIndirVD/data/HSrats_noHaplotypes.hdf5'
        f = h5py.File(in_file,'r')

        # read trait and covariantes 
        trait = 'Distance0_30_bc'
        measures = f['phenotypesNcovariates']['cols_measures']['measures'][:]
        Ip = measures==trait
        covs = f['phenotypesNcovariates']['cols_measures']['covariates2use'][Ip][0].split(',')
        Ic = sp.zeros(Ip.shape[0],dtype=bool)
        for cov in covs:    Ic = sp.logical_or(Ic,measures==cov)
        Y = f['phenotypesNcovariates']['array'][Ip,:].T
        cov = f['phenotypesNcovariates']['array'][Ic,:].T
        Is = sp.logical_and((cov!=-999).all(1),Y[:,0]!=-999)
        Y = Y[Is,:]; cov = cov[Is,:]
        cage = f['phenotypesNcovariates']['rows_subjects']['cage'][Is]
        sampleID = f['phenotypesNcovariates']['rows_subjects']['outbred'][Is]
        #TODO: gaussianize phenotype?

        # grab kinship
        k_sampleID = f['kinships']['genotypes_IBS']['cols_subjects']['outbred'][:]
        f['kinships']['genotypes_IBS']['array']

        ipdb.set_trace()

    # define model and optimize
    vc = DirIndirVD(pheno=Y, kinship=kinship, cage=cage)
    vc.optimize()

    ipdb.set_trace()


