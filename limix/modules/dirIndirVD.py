import h5py
import scipy as sp
from limix.core.gp.gp_base import GP
from limix.core.mean.mean_base import mean_base as lin_mean
from limix.core.covar.dirIndirCov import DirIndirCov
from limix.core.covar.fixed import FixedCov 
from limix.core.covar.combinators import SumCov
from limix.utils.preprocess import covar_rescaling_factor
from limix.utils.preprocess import covar_rescale
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

        # rescaling of covariances
        kinship = covar_rescale(kinship)
        WW = covar_rescale(WW)
        _ZKZ = sp.dot(Z,sp.dot(kinship,Z))
        _ZZ  = sp.dot(Z,Z)
        sf_Zg = sp.sqrt(covar_rescaling_factor(_ZKZ))
        sf_Ze = sp.sqrt(covar_rescaling_factor(_ZZ))
        Zg = sf_Zg * Z
        Ze = sf_Ze * Z

        # define mean
        self.mean = lin_mean(pheno,covs)

        # define covariance matrices
        self._genoCov = DirIndirCov(kinship,Z)
        self._envCov = DirIndirCov(sp.eye(self.N),Z)
        self._cageCov = FixedCov(WW)
        covar = SumCov(self._genoCov,self._envCov,self._cageCov)

        #self._genoCov.setRandomParams()
        #print self._genoCov.covff.K_grad_interParam_i(0)
        #print self._genoCov.covff.K_grad_interParam_i(1)
        #print self._genoCov.covff.K_grad_interParam_i(2)

        # define gp
        self._gp = GP(covar=covar,mean=self.mean)

    def optimize(self, calc_ste = False, verbose = True):
        if 0:
            # trial for inizialization it is complicated though
            cov = sp.array([[0.2,1e-4],[1e-4,1e-4]])
            self._genoCov.setCovariance(cov)
            self._envCov.setCovariance(cov)
            self._cageCov.scale = 0.2
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
        R['var_C'] = self._cageCov.scale

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
        K = self._envCov.K() + self._cageCov.K()
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
        return self._cageCov.scale

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

    # TODO: beautify the output and stuff

    # define model and optimize
    vc = DirIndirVD(pheno=Y, kinship=kinship, cage=cage, covs = covs)
    rv = vc.optimize(calc_ste = True)
    print 'lml:', rv['LML']
    res = vc.getResidual()
    print 'residual:', res['variance_explained']
    print 'geno cov'
    print vc._genoCov.dirIndirCov_K()
    print 'ste on geno cov'
    print vc._genoCov.dirIndirCov_K_ste()
