"""GP testing code"""
import unittest
import scipy as SP
import pdb
import limix
import scipy.linalg as linalg


def PCA(Y, components):
    """run PCA, retrieving the first (components) principle components
    return [s0, eig, w0]
    s0: factors
    w0: weights
    """
    sv = linalg.svd(Y, full_matrices=0);
    [s0, w0] = [sv[0][:, 0:components], SP.dot(SP.diag(sv[1]), sv[2]).T[:, 0:components]]
    v = s0.std(axis=0)
    s0 /= v;
    w0 *= v;
    return [s0, w0]



class CGPLVM_test(unittest.TestCase):
    """oGPLVM test class"""

    def simulate(self):
        """simulate a dataset. Note this is seed-dependent"""

        N = self.settings['N']
        K = self.settings['K']
        D = self.settings['D']

        SP.random.seed(1)
        S = SP.random.randn(N,K)
        W = SP.random.randn(D,K)

        Y = SP.dot(W,S.T).T
        Y+= 0.1*SP.random.randn(N,D)

        X0 = SP.random.randn(N,K)
        X0 = PCA(Y,K)[0]
        RV = {'X0': X0,'Y':Y,'S':S,'W':W}
        return RV
        
    def setUp(self):
        SP.random.seed(1)

        #1. simulate
        self.settings = {'K':5,'N':100,'D':80}
        self.simulation = self.simulate()

        N = self.settings['N']
        K = self.settings['K']
        D = self.settings['D']

        #2. setup GP
        covar  = limix.CCovLinearISO(K)
        ll  = limix.CLikNormalIso()
        #create hyperparm     
        covar_params = SP.array([1.0])
        lik_params = SP.array([1.0])
        hyperparams = limix.CGPHyperParams()
        hyperparams['covar'] = covar_params
        hyperparams['lik'] = lik_params
        hyperparams['X']   = self.simulation['X0']
        #cretae GP
        self.gp=limix.CGPbase(covar,ll)
        #set data
        self.gp.setY(self.simulation['Y'])
        self.gp.setX(self.simulation['X0'])
        self.gp.setParams(hyperparams)
        pass

    
    def test_fit(self):
        #create optimization object
        self.gpopt = limix.CGPopt(self.gp)
        #run
        RV = self.gpopt.opt()
        RV = self.gpopt.opt()
        
        RV = RV & (SP.absolute(self.gp.LMLgrad()['X']).max()<1E-1)
        RV = RV & (SP.absolute(self.gp.LMLgrad()['covar']).max()<1E-1)
        RV = RV & (SP.absolute(self.gp.LMLgrad()['lik']).max()<1E-1)
        self.assertTrue(RV)


class CGPLVM_test_constK(CGPLVM_test):
    """adapted version of GPLVM test, including a fixed CF covaraince"""

    def setUp(self):
        SP.random.seed(1)

        #1. simulate
        self.settings = {'K':5,'N':100,'D':80}
        self.simulation = self.simulate()

        N = self.settings['N']
        K = self.settings['K']
        D = self.settings['D']

        #2. setup GP        
        K0 = SP.dot(self.simulation['S'],self.simulation['S'].T)
        K0[:] = 0

        covar1 = limix.CFixedCF(K0)
        covar2 = limix.CCovLinearISO(K)
        covar  = limix.CSumCF()
        covar.addCovariance(covar1)
        covar.addCovariance(covar2)
         
        ll  = limix.CLikNormalIso()
        #create hyperparm     
        covar_params = SP.array([0.0,1.0])
        lik_params = SP.array([0.1])
        hyperparams = limix.CGPHyperParams()
        hyperparams['covar'] = covar_params
        hyperparams['lik'] = lik_params
        hyperparams['X']   = self.simulation['X0']
        #cretae GP
        self.gp=limix.CGPbase(covar,ll)
        #set data
        self.gp.setY(self.simulation['Y'])
        self.gp.setX(self.simulation['X0'])
        self.gp.setParams(hyperparams)
        pass



if __name__ == '__main__':
    unittest.main()


