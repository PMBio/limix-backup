import scipy as sp
from limix.core.gp.gp_base import GP
from limix.core.mean.mean_base import mean_base as lin_mean
from limix.core.covar.dirIndirCov import DirIndirCov
from limix.core.covar.fixed import FixedCov 
from limix.core.covar.combinators import SumCov
import ipdb

class DirIndirVD():

    def __init__(self, pheno = None, kinship = None, design = None, covs=None):

        assert pheno is not None, 'Specify pheno!'
        assert kinship is not None, 'Specify kinship!'
        assert design is not None, 'Specify design!'

        if covs is None:
            covs = sp.ones((n,1)) 

        self.N = pheno.shape[0]

        # define mean
        mean = lin_mean(pheno,covs)

        # define covariance matrices
        self._genoCov = DirIndirCov(kinship,design)
        self._envCov = DirIndirCov(sp.eye(self.N),design)
        self._noisCov = FixedCov(sp.eye(self.N))
        covar = SumCov(self._genoCov,self._envCov,self._noisCov)

        # define gp
        self._gp = GP(covar=covar,mean=mean)

    def optimize(self, init_params = 'standard'):

        if 0:
            # trial for inizialization it is complicated though
            cov = sp.array([[0.2,1e-4],[1e-4,1e-4]])
            self._genoCov.setCovariance(cov)
            self._envCov.setCovariance(cov)
            self._noisCov.scale = 0.2
        else:
            self._gp.covar.setRandomParams()

        ipdb.set_trace()
        self._gp.optimize()

    def getDirIndirGenoCovar(self):
        return self._genoCov.dirIndirCov_K()

    def getDirIndirEnvCovar(self):
        return self._envCov.dirIndirCov_K()

    def getDirInderNoisVar(self):
        return self._noisCov.scale

if __name__=='__main__':

    # generate data
    n = 10
    f = 2
    X  = 1.*(sp.rand(n,f)<0.2)
    X -= X.mean(0); X /= X.std(0)
    kinship  = sp.dot(X,X.T)
    kinship /= kinship.diagonal().mean()
    design = sp.zeros((n,n))
    for i in range(n/2):
        design[2*i,2*i+1] = 1
        design[2*i+1,2*i] = 1
    Y = sp.randn(n,1)

    # define model and optimize
    vc = DirIndirVD(pheno=Y, kinship=kinship, design=design)
    vc.optimize()

    ipdb.set_trace()


