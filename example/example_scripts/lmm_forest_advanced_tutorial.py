'''
Created on Sep 25, 2013

@author: johannes
'''


import scipy as SP
from mixed_forest.MixedForest import Forest as LMF
import pylab as PL
import mixedForestUtils as utils

if __name__ == '__main__':
    #######################
    ### data simulation ###
    #######################
    # we simulate genetric features as X as binary encoding of integer numbers
    SP.random.seed(42)
    n_samples=2**8
    x = SP.arange(n_samples).reshape(-1,1)
    X = utils.convertToBinaryPredictor(x)
    y_fixed = X[:,0:1] * X[:,2:3]
    kernel=utils.getQuadraticKernel(x, d=200)
    y_conf = y_fixed.copy()
    y_conf += SP.random.multivariate_normal(SP.zeros(n_samples),kernel).reshape(-1,1)
    y_conf += .1*SP.random.randn(n_samples,1)
    (training, test) = utils.crossValidationScheme(2, n_samples)
    SP.random.seed(42)
    lm_forest = LMF(kernel=kernel[SP.ix_(training, training)], sampsize=.5, verbose=1, n_estimators=100)
    lm_forest.fit(X[training], y_conf[training])
    response_tot = lm_forest.predict(X[test], kernel[SP.ix_(test,training)])
    # make random forest prediction for comparision
    random_forest = LMF(kernel='iid')
    random_forest.fit(X[training], y_conf[training])
    response_iid = random_forest.predict(X[test])
    response_fixed = lm_forest.predict(X[test])
    PL.plot(x, y_fixed, 'g--')
    PL.plot(x, y_conf, '.7')
    PL.plot(x[test], response_tot, 'r-.')
    PL.plot(x[test], response_fixed, 'c-.')
    PL.plot(x[test], response_iid, 'b-.')
    PL.title('prediction')
    PL.xlabel('genotype (in decimal encoding)')
    PL.ylabel('phenotype')
    PL.show()
    PL.close()
    feature_scores_lmf = lm_forest.log_importance
    feature_scores_rf = random_forest.log_importance
    n_predictors = X.shape[1]
    PL.bar(SP.arange(n_predictors), feature_scores_lmf, .3, color='r')
    PL.bar(SP.arange(n_predictors)+.3, feature_scores_rf, .3, color='b')
    PL.title('feature importances')
    PL.xlabel('feature dimension')
    PL.ylabel('log feature score')
    PL.xticks(SP.arange(n_predictors)+.3, SP.arange(n_predictors)+1)
    PL.legend(['linear mixed forest', 'random forest'])
    PL.show()
