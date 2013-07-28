"""GP testing code"""
import scipy as SP
import scipy.stats
import pdb
import sys
import limix

sys.path.append('./../helper/')
from helper import message

class gpbase_test:
    """test class for CVarianceDecomposition"""
    
    def __init__(self):
        self.generate()
    
    def genY(self):
        self.y=SP.random.multivariate_normal(SP.zeros(self.n_samples),(self.gp.getCovar()).K()+(self.gp.getLik()).K())
    
    def generate(self):
        SP.random.seed(1)
        self.n_dimensions=2
        self.n_samples = 100
        X = SP.rand(self.n_samples,self.n_dimensions)
        covar  = limix.CCovSqexpARD(self.n_dimensions)
        ll  = limix.CLikNormalIso()
        covar_params = SP.array([1,1,1])
        lik_params   = SP.array([0.5])
        hyperparams0 = limix.CGPHyperParams()
        hyperparams0['covar'] = covar_params
        hyperparams0['lik'] = lik_params
        self.gp=limix.CGPbase(covar,ll)
        self.gp.setX(X)
        self.gp.setParams(hyperparams0)
        self.genY()
        self.gp.setY(self.y)
    
    def test_fit(self):
        #create optimization object
        self.gpopt = limix.CGPopt(self.gp)
        #run
        self.gpopt.opt()
        params = SP.concatenate((self.gp.getParams()['covar'],self.gp.getParams()['lik']))[:,0]
        params_true = SP.array([0.28822188,  0.35271548,  0.13709146,  0.49447424])
        RV = (params-params_true).max()<1e-6
        print '   ...fit %s' % message(RV)
    
    def test_all(self):
        print '... testing GP'
        self.test_fit()


if __name__ == '__main__':
    testGP = gpbase_test()
    testGP.test_all()


