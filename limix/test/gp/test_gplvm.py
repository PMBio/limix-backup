"""GP testing code"""
import unittest
import scipy as SP
import numpy as np
import limix.deprecated as dlimix
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
        covar  = dlimix.CCovLinearISO(K)
        ll  = dlimix.CLikNormalIso()
        #create hyperparm
        covar_params = SP.array([1.0])
        lik_params = SP.array([1.0])
        hyperparams = dlimix.CGPHyperParams()
        hyperparams['covar'] = covar_params
        hyperparams['lik'] = lik_params
        hyperparams['X']   = self.simulation['X0']
        #cretae GP
        self.gp=dlimix.CGPbase(covar,ll)
        #set data
        self.gp.setY(self.simulation['Y'])
        self.gp.setX(self.simulation['X0'])
        self.gp.setParams(hyperparams)
        pass

    @unittest.skip("someone has to fix it")
    def test_fit(self):
        #create optimization object
        self.gpopt = dlimix.CGPopt(self.gp)
        #run
        RV = self.gpopt.opt()
        RV = self.gpopt.opt()

        m = (SP.absolute(self.gp.LMLgrad()['X']).max() +
             SP.absolute(self.gp.LMLgrad()['covar']).max() +
             SP.absolute(self.gp.LMLgrad()['lik']).max())

        np.testing.assert_almost_equal(m, 0., decimal=1)


class CGPLVM_test_constK(unittest.TestCase):
    """adapted version of GPLVM test, including a fixed CF covaraince"""

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
        K0 = SP.dot(self.simulation['S'],self.simulation['S'].T)
        K0[:] = 0

        covar1 = dlimix.CFixedCF(K0)
        covar2 = dlimix.CCovLinearISO(K)
        covar  = dlimix.CSumCF()
        covar.addCovariance(covar1)
        covar.addCovariance(covar2)

        ll  = dlimix.CLikNormalIso()
        #create hyperparm
        covar_params = SP.array([0.0,1.0])
        lik_params = SP.array([0.1])
        hyperparams = dlimix.CGPHyperParams()
        hyperparams['covar'] = covar_params
        hyperparams['lik'] = lik_params
        hyperparams['X']   = self.simulation['X0']
        #cretae GP
        self.gp=dlimix.CGPbase(covar,ll)
        #set data
        self.gp.setY(self.simulation['Y'])
        self.gp.setX(self.simulation['X0'])
        self.gp.setParams(hyperparams)
        pass

    @unittest.skip("someone has to fix it")
    def test_fit(self):
        #create optimization object
        self.gpopt = dlimix.CGPopt(self.gp)
        #run
        RV = self.gpopt.opt()
        RV = self.gpopt.opt()

        m = (SP.absolute(self.gp.LMLgrad()['X']).max() +
             SP.absolute(self.gp.LMLgrad()['covar']).max() +
             SP.absolute(self.gp.LMLgrad()['lik']).max())

        np.testing.assert_almost_equal(m, 0., decimal=1)


if __name__ == '__main__':
    unittest.main()
