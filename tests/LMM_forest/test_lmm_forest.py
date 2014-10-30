'''
Created on Sep 19, 2013

@author: johannes
'''
# create some test cases
import scipy as SP
import limix.modules.mixedForestUtils as utils
from limix.modules.lmm_forest import Forest as MF
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import metrics
from sklearn import cross_validation as CV
import pylab as PL
import unittest


class TestMixedForest(unittest.TestCase):

    def setUp(self, n=100, m=1):
        SP.random.seed(1)
        self.x, self.y = utils.lin_data_cont_predictors(n=n,m=m)
        self.n, self.m = self.x.shape
        [self.train, self.test] = utils.crossValidationScheme(2,self.n)
        self.n_estimators = 100

    def test_delta_updating(self):
        n_sample = 100
        # A 20 x 2 random integer matrix
        X = SP.empty((n_sample, 2))
        X[:, 0] = SP.arange(0, 1, 1.0/n_sample)
        X[:, 1] = SP.random.rand(n_sample)
        sd_noise = .5
        sd_conf = .5
        noise = SP.random.randn(n_sample, 1)*sd_noise

        print 'true delta equals', (sd_noise**2)/(sd_conf**2)
        # Here, the observed y is just a linear function of the first column
        # in X and # a little independent gaussian noise
        y_fixed = (X[:, 0:1] > .5)*1.0
        y_fn = y_fixed + noise

        # Divide into training and test sample using 2/3 of data for training
        training_sample = SP.zeros(n_sample, dtype='bool')
        training_sample[
            SP.random.permutation(n_sample)[:SP.int_(.66*n_sample)]] = True
        test_sample = ~training_sample

        kernel = utils.getQuadraticKernel(X[:, 0], d=0.0025)
        # The confounded version of y_lin is computed as
        y_conf = sd_conf*SP.random.multivariate_normal(SP.zeros(n_sample),
                                                       kernel, 1).reshape(-1, 1)
        y_tot = y_fn + y_conf
        # Selects rows and columns
        kernel_train = kernel[SP.ix_(training_sample, training_sample)]
        kernel_test = kernel[SP.ix_(test_sample, training_sample)]
        lm_forest = MF(kernel=kernel_train, update_delta=False, max_depth=1,
                       verbose=0)
        # Returns prediction for random effect
        lm_forest.fit(X[training_sample], y_tot[training_sample])
        response_lmf = lm_forest.predict(X[test_sample], k=kernel_test)

        print 'fitting forest (delta-update)'
        # earn random forest, not accounting for the confounding
        random_forest = MF(kernel=kernel_train, update_delta=True, max_depth=5,
                           verbose=0)
        random_forest.fit(X[training_sample], y_tot[training_sample])
        response_rf = random_forest.predict(X[test_sample], k=kernel_test)


    def test_sklearn_crossvalidation_kernel(self):
        self.setUp(n=100, m=110)
        kernel = SP.ones(self.m, dtype='bool')
        kernel[:10] = False
        mf = MF(n_estimators=self.n_estimators, kernel=kernel)
        scores = CV.cross_val_score(mf, self.x, self.y, scoring='r2')
        self.assertTrue(SP.absolute((scores[0] - scores[1]) < 0.3))

    def test_sklearn_crossvalidation(self):
        self.setUp(n=100, m=10)
        mf = MF(n_estimators=self.n_estimators, kernel='iid')
        scores = CV.cross_val_score(mf, self.x, self.y, scoring='r2')
        self.assertTrue(SP.absolute((scores[0] - scores[1]) < 0.1))
        pass

    def test_kernel_builing(self):
        SP.random.seed(42)
        X = (SP.random.rand(5, 10) > .5)*1.0
        kernel = utils.estimateKernel(X, scale=False)
        small_kernel = utils.estimateKernel(X[:, 0:5], scale=False)
        small_kernel_test = utils.update_Kernel(kernel, X[:, 5:], scale=False)
        self.assertAlmostEqual((small_kernel -
                                small_kernel_test).sum(), 0, places=14)

    def test_consistency_cont_cpp(self):
        # Test: consistency to rf on cont. predictors (cpp)
        model = MF(kernel=SP.identity(self.train.sum()),
                   n_estimators=self.n_estimators, cpp_fit=True,
                   cpp_predict=True, subsampling=False, min_samples_split=2)

        model.fit(self.x[self.train], self.y[self.train])
        response = model.predict(self.x[self.test])
        rf = RandomForestRegressor(n_estimators=self.n_estimators)
        rf.fit(self.x[self.train], self.y[self.train].reshape(-1))
        response_rf = rf.predict(self.x[self.test])
        self.assertTrue(SP.absolute(response - response_rf).sum() < 2400)

    def test_consistency_cont_py(self):
        # Test: consistency to rf on cont. predictors (py)
        model = MF(kernel=SP.identity(self.train.sum()),
                   n_estimators=self.n_estimators,
                   cpp_fit=False, cpp_predict=False, subsampling=False,
                   min_samples_split=2)
        model.fit(self.x[self.train], self.y[self.train])
        response = model.predict(self.x[self.test])
        rf = RandomForestRegressor(n_estimators=self.n_estimators)
        rf.fit(self.x[self.train], self.y[self.train].reshape(-1))
        response_rf = rf.predict(self.x[self.test])
        self.assertTrue(SP.absolute(response - response_rf).sum() < 2400)

    def test_consistency_cont_multi_dim(self):
        # Test: consistency to rf on cont. predictors (cpp)
        self.setUp(m=10)
        model = MF(kernel=SP.identity(self.train.sum()),
                   n_estimators=self.n_estimators, cpp_fit=True,
                   cpp_predict=True, subsampling=False, min_samples_split=2)
        model.fit(self.x[self.train], self.y[self.train])
        response = model.predict(self.x[self.test])
        rf = RandomForestRegressor(n_estimators=self.n_estimators)
        rf.fit(self.x[self.train], self.y[self.train].reshape(-1))
        response_rf = rf.predict(self.x[self.test])
        self.assertTrue(SP.absolute(response - response_rf).sum() < 2400)

    def test_consistency_cont_multi_dim_big(self):
        # Test: consistency to rf on cont. predictors (cpp)
        self.setUp(m=100)
        SP.random.seed(10)
        self.n_estimators = 100
        model = MF(kernel=SP.identity(self.train.sum()),
                   n_estimators=self.n_estimators, subsampling=False,
                   min_samples_split=2, ratio_features=1)
        model.fit(self.x[self.train], self.y[self.train])
        response = model.predict(self.x[self.test])
        rf = RandomForestRegressor(n_estimators=self.n_estimators)
        rf.fit(self.x[self.train], self.y[self.train].reshape(-1))
        response_rf = rf.predict(self.x[self.test])
        rf.fit(self.x[self.train], self.y[self.train].reshape(-1))
        self.assertTrue(SP.absolute(response - response_rf).sum() < 2400)


    def test_depth_building(self):
        self.setUp(m=10)
        X = self.x.copy()
        X -= X.mean(axis=0)
        X /= X.std(axis=0)
        kernel = SP.dot(X, X.T)
        train = SP.where(self.train)[0]
        test = SP.where(~self.train)[0]
        model = MF(fit_optimal_depth=True, max_depth=3,
                   kernel=kernel[SP.ix_(train, train)])
        model.fit(self.x[self.train], self.y[self.train],
                  fit_optimal_depth=True)
        prediction_1 = model.predict(X[test], k=kernel[test, train],
                                     depth=model.opt_depth)
        # Grow to end
        model.further()
        # Prediction again
        prediction_2 = model.predict(X[test], k=kernel[test, train],
                                     depth=model.opt_depth)
        self.assertEqual((prediction_1 - prediction_2).sum(), 0.0)

    def test_depth_building_linear(self):
        SP.random.seed(42)
        n_samples = 2**8
        x = SP.arange(n_samples).reshape(-1, 1)
        X = utils.convertToBinaryPredictor(x)
        y_fixed = X[:, 0:1] * X[:, 2:3]
        kernel = utils.getQuadraticKernel(x, d=200)
        y_conf = y_fixed.copy()
        y_conf += SP.random.multivariate_normal(SP.zeros(n_samples),
                                                kernel, 1).reshape(-1, 1)
        y_conf += .1*SP.random.randn(n_samples, 1)
        (training, test) = utils.crossValidationScheme(2, n_samples)
        SP.random.seed(42)
        lm_forest = MF(kernel=kernel[SP.ix_(training, training)], sampsize=.5,
                       n_estimators=100, fit_optimal_depth=True)
        lm_forest.fit(X[training], y_conf[training])
        self.assertEqual(2, lm_forest.opt_depth, msg='fitting of optimal depth')

        response_tot = lm_forest.predict(X[test],
                                         kernel[SP.ix_(test, training)])
        random_forest = MF(kernel='iid')
        random_forest.fit(X[training], y_conf[training])
        response_iid = random_forest.predict(X[test])
        diff_rf_mf = (response_tot - response_iid).sum()
        self.assertAlmostEqual(diff_rf_mf, 7.07, 2, msg='difference rf and mf')

        diff_mf = (response_tot - SP.zeros_like(response_tot)).sum()
        self.assertAlmostEqual(diff_mf, 2.75, 2, msg='mf correctness')
        self.assertEqual(2, lm_forest.opt_depth, msg='fitting of optimal depth')

    def test_forest_stump_recycling(self):
        self.setUp(m=5)
        SP.random.seed(42)
        model = MF(fit_optimal_depth=True, kernel='iid',
                   build_to_opt_depth=True)
        model.fit(self.x[self.train], self.y[self.train])
        prediction_1 = model.predict(self.x[self.test], depth=model.opt_depth)
        model.fit(self.x[self.train], self.y[self.train], recycle=True)
        prediction_2 = model.predict(self.x[self.test], depth=model.opt_depth)
        self.assertGreater(.7, ((prediction_1 - prediction_2)**2).sum())

    def test_normalization_kernel(self):
        SP.random.seed(42)
        n = 50
        m = 100
        X = (SP.random.rand(n, m) > .5)*1.
        X_test = (SP.random.rand(10, m) > .5)*1.
        K = utils.estimateKernel(X)
        y = SP.random.rand(n, 1)
        SP.random.seed(1)
        mf = MF(kernel=K)
        mf.fit(X, y)
        results_1 = mf.predict(X_test)

        X -= X.mean(axis=0)
        X /= X.std(axis=0)
        X_test -= X_test.mean(axis=0)
        X_test /= X_test.std(axis=0)

        SP.random.seed(1)
        mf = MF(kernel=K)
        mf.fit(X, y)
        results_2 = mf.predict(X_test)
        self.assertEqual(results_1.sum(), results_2.sum())

    def polynom(self, x):
        return -x + x**3

    def complete_sample(self, x, mean=0, var=.3**2):
        return self.polynom(x) + SP.random.randn(x.size) * SP.sqrt(var) + mean

    def test_covariate_shift(self):
        n_sample = 100
        # Biased training
        var_bias = .5**2
        mean_bias = .7
        x_train = SP.random.randn(n_sample)*SP.sqrt(var_bias) + mean_bias
        y_train = self.complete_sample(x_train)

        # Unbiased test set
        var = .3**2
        mean = 0

        x_test = SP.random.randn(n_sample)*SP.sqrt(var) + mean
        x_complete = SP.hstack((x_train, x_test))

        kernel = utils.getQuadraticKernel(x_complete, d=1) +\
            10 * SP.dot(x_complete.reshape(-1, 1), x_complete.reshape(1, -1))
        kernel = utils.scale_K(kernel)
        kernel_train = kernel[SP.ix_(SP.arange(x_train.size),
                                     SP.arange(x_train.size))]
        kernel_test = kernel[SP.ix_(SP.arange(x_train.size, x_complete.size),
                             SP.arange(x_train.size))]

        mf = MF(n_estimators=100, kernel=kernel_train, min_depth=0,
                subsampling=False)
        mf.fit(x_train.reshape(-1, 1), y_train.reshape(-1, 1))
        response_gp = mf.predict(x_test.reshape(-1, 1), kernel_test, depth=0)
        self.assertTrue(((response_gp - self.polynom(x_test))**2).sum() < 2.4)

if __name__ == '__main__':
    unittest.main()
