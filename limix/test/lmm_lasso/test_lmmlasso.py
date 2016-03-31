"""Variance Decomposition testing code"""
import unittest
import scipy as SP
import numpy as np
import scipy.stats
import pdb
import os
import sys
import limix.deprecated as dlimix
import limix.deprecated.modules.lmmlasso as lmmlasso
from limix.test import data


class Lmmlasso_test(unittest.TestCase):
    """test class for lmm-lasso
    """

    def genGeno(self):
        X  = (SP.rand(self.N,self.S)<0.2)*1.
        X -= X.mean(0)
        X /= X.std(0)
        self.D['X'] = X

    def genKernel(self):
        X  = (SP.rand(self.N,10)<0.2)*1.
        K  = SP.dot(X,X.T)
        K /= SP.diag(K).mean()
        K += 1e-3*SP.eye(self.N)
        self.D['K'] = K


    def genPheno(self):

        idx_causal = SP.random.randint(0,self.S,10)
        sigma_g = 0.25
        sigma_e = 0.25
        sigma_f = 0.50

        u = SP.random.multivariate_normal(SP.zeros(self.N),self.D['K'])
        u*= SP.sqrt(sigma_g)/u.std()

        e = SP.random.randn(self.N)
        e*= SP.sqrt(sigma_e)/e.std()

        f = SP.sum(self.D['X'][:,idx_causal],axis=1)
        f*= SP.sqrt(sigma_f)/f.std()

        y = u + e + f
        self.D['y']= y
        self.D['causal_idx'] = idx_causal




    def setUp(self):
        #check: do we have a csv File?
        self.dir_name = os.path.dirname(os.path.realpath(__file__))
        self.dataset = os.path.join(self.dir_name,'lmmlasso')

        if (not os.path.exists(self.dataset)) or 'recalc' in sys.argv:
            if not os.path.exists(self.dataset):
                os.makedirs(self.dataset)
            SP.random.seed(1)
            self.N = 500
            self.S = 100
            self.D = {}
            self.genGeno()
            self.genKernel()
            self.genPheno()
            self.generate = True
        else:
            self.generate=False
            self.D = data.load(self.dataset)
            self.N = self.D['X'].shape[0]
            self.S = self.D['X'].shape[1]

        self.lmmlasso = lmmlasso.LmmLasso()


    def test_fit(self):
        """ test fitting """
        self.lmmlasso.set_params(alpha=1e-1)
        self.lmmlasso.fit(self.D['X'],self.D['y'],self.D['K'])
        params = self.lmmlasso.coef_
        yhat = self.lmmlasso.predict(self.D['X'],self.D['K'])

        if self.generate:
            self.D['params_true'] = params
            self.D['yhat'] = yhat
            data.dump(self.D,self.dataset)
            self.generate=False
        params_true = self.D['params_true']
        yhat_true   = self.D['yhat']

        RV = ((SP.absolute(params)-SP.absolute(params_true))**2).max()
        np.testing.assert_almost_equal(RV, 0., decimal=4)

        RV = ((SP.absolute(yhat)-SP.absolute(yhat_true))**2).max()
        np.testing.assert_almost_equal(RV, 0., decimal=2)



if __name__ == '__main__':
    unittest.main()
