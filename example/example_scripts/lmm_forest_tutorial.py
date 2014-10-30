'''
Created on Sep 24, 2013

@author: johannes
'''
import scipy as SP
import pylab as PL # plotting
import mixedForestUtils as utils
from mixed_forest.MixedForest import Forest as LMF # linear mixed forest class


if __name__ == '__main__':
    SP.random.seed(43)
    # first of all, create some simple data set including a singe fixed effect
    n_sample = 100
    X = SP.empty((n_sample,2)) # a 20 x 2 random integer matrix
    X[:,0] = SP.arange(0,1,1.0/n_sample) 
    X[:,1] = SP.random.rand(n_sample)
    noise = SP.random.randn(n_sample,1)*.05
    # here, the observed y is just a linear function of the first column in X and 
    # a little independent gaussian noise
    y_fixed = (X[:,0:1] > .5)*.5
    y_fn  = y_fixed +  noise
    
    # now we make it a bit more interesting, and add a random effect which makes our samples dependent# 
    kernel = utils.getQuadraticKernel(X[:,0], d=0.0025)
    # the confounded version of y_lin is computed as
    y_conf = .5*SP.random.multivariate_normal(SP.zeros(n_sample), kernel).reshape(-1,1)
    y_tot = y_fn + y_conf
    
    # divide into training and test sample using 2/3 of data for training
    training_sample = SP.zeros(n_sample, dtype='bool')
    training_sample[SP.random.permutation(n_sample)[:SP.int_(.66*n_sample)]] = True
    test_sample = ~training_sample
    
    kernel = utils.getQuadraticKernel(X[:,0], d=0.0025)
    # the confounded version of y_lin is computed as
    y_conf = .5*SP.random.multivariate_normal(SP.zeros(n_sample), kernel).reshape(-1,1)
    y_tot = y_fn + y_conf

    PL.plot(X[:,0], y_fn, 'c')
    PL.plot(X[:,0], y_conf+y_fn, 'm')
    PL.plot(X[:,0], y_conf + y_fixed, 'g')
    PL.legend(['fixed effect + ind. noise', 'fixed effect + confounding + ind. noise', 'fixed effect + confounding'], loc=2)
    PL.show()
    PL.close()
    
    #learn random forest, not accounting for the confounding
    random_forest = LMF(kernel='iid')
    random_forest.fit(X[training_sample],y_tot[training_sample])
    response_rf = random_forest.predict(X[test_sample])

    
    kernel_train = kernel[SP.ix_(training_sample, training_sample)] #selects rows and columns
    kernel_test =  kernel[SP.ix_(test_sample, training_sample)]
    lm_forest = LMF(kernel=kernel_train)
    lm_forest.fit(X[training_sample],y_tot[training_sample])
    response_lmf = lm_forest.predict(X[test_sample], k=kernel_test) #returns prediction for random effect
    
    PL.plot(X[:,0:1], y_fixed + y_conf, 'g')
    PL.plot(X[test_sample,0:1], response_lmf, '.b-.')
    PL.plot(X[test_sample,0:1], response_rf, '.k-.')
    PL.ylabel('predicted y')
    PL.xlabel('first dimension of X')
    PL.legend(['ground truth (without ind. noise)', 'linear mixed forest', 'random forest'], loc=2)
    PL.show()
