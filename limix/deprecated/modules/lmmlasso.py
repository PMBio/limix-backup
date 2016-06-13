'''
Created on Dec 19, 2013

@author: johannes stephan, barbara rakitsch
'''

from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
import sklearn.cross_validation as cross_validation

from .varianceDecomposition import VarianceDecomposition
from . import qtl
import scipy as SP
import numpy as NP
import scipy.stats as ST
import time
import scipy.linalg as LA
import scipy.optimize as OPT
import pdb


def compute_linear_kernel(X,idx=None,jitter=1e-3,standardize=True):
    """
    compute linear kernel

    X           : SNP data [N x F]
    idx         : boolean vector of size F, indicating which SNPs to use to build the covariance matrix
    standardize : if True (default), covariance matrix is standardized to unit variance
    jitter      : adds jitter to the diagonal of the covariance matrix (default: 1e-3)
    """
    N = X.shape[0]
    if idx is not None:
        K = SP.dot(X[:,idx],X[:,idx].T)
    else:
        K = SP.dot(X,X.T)

    if standardize:
        K /= SP.diag(K).mean()

    if jitter:
        K += jitter*SP.eye(N)

    return K


def runStabilitySelection(estimator,X,y,K=None,n_repeats=100,frac_sub=0.5,verbose=False):
    """
    estimator   : (Lmm-)Lasso model
    X           : SNP data
    y           : phenotype vector
    K           : background covariance matrix
    n_repeats   : number of repetitions
    frac_sub    : size of the subsample
    """
    N,F = X.shape
    Nsub = int(frac_sub*N)

    W_nonzero = SP.zeros((n_repeats,F),dtype=bool)

    for irep in range(n_repeats):
        if verbose: print(('Running %d/%d'%(irep,n_repeats)))
        iperm = SP.random.permutation(N)[:Nsub]

        if K is not None:
            estimator.fit(X[iperm],y[iperm],K=K[iperm][:,iperm])
        else:
            estimator.fit(X[iperm],y[iperm])

        W_nonzero[irep] = estimator.coef_!=0

    p_stab = W_nonzero.mean(0)
    return p_stab


def testLassoBackgroundModel(estimator,Xfg,Xbg,y,use_Kgeno=True,**kwargs):
    """
    run a linear mixed model on the foreground SNPs (Xfg), while estimating SNPs for the background covariance matrix via the Lasso usting the background SNPs(Xbg).

    estimator   : (Lmm-)Lasso model
    Xbg         : SNP data for the background model
    Xfg         : SNP data for the foreground model
    y           : phenotype vector
    use_Kgene   : if True (default), background covariance matrix is estimated and used.
    """
    if use_Kgeno:
        Kbg = compute_linear_kernel(Xbg)
        estimator.fit(Xfg,y,Kbg)
        iactive = estimator.coef_!=0

        if iactive.any():
            Kfg = compute_linear_kernel(Xbg,iactive)
            vd = VarianceDecomposition(y)
            vd.addRandomEffect(is_noise=True)
            vd.addRandomEffect(Kbg)
            vd.addRandomEffect(Kfg)
            vd.optimize()
            K = vd.gp.getCovar().K()
        else:
            K = Kbg

    else:
        estimator.fit(Xbg,y)
        idx = estimator.coef_!=0
        K   = compute_linear_kernel(Xbg,idx)

    lmm = qtl.test_lmm(Xfg,y,K=K,**kwargs)
    pv  = lmm.getPv()

    return pv

def runCrossValidation(estimator,X,y,alphas,K=None,n_folds=10,verbose=False):
    """ run cross-validation

    estimator   : (Lmm-)Lasso model
    X           : SNP data
    y           : phenotype vector
    alphas      : list of l1-regularization parameter
    K           : background covariance matrix
    n_folds     : number of folds used in cross-validation
    """

    """ setting up """
    N = X.shape[0]
    kfold = cross_validation.KFold(N, n_folds=n_folds)
    n_alphas = len(alphas)
    MSE_train = SP.zeros((n_folds,n_alphas))
    MSE_test  = SP.zeros((n_folds,n_alphas))
    W_nonzero = SP.zeros((n_folds,n_alphas))

    t1 = time.time()
    ifold = 0
    for train,test in kfold:
        if verbose: print(('running fold %d'%ifold))

        """ splitting into training and testing """
        X_train = X[train]
        X_test  = X[test]
        y_train = y[train]
        y_test  = y[test]
        if K is not None:
            K_train = K[train][:,train]
            K_test  = K[test][:,train]

        for ialpha in range(n_alphas):
            estimator.set_params(alpha=alphas[ialpha])
            if K is None:
                estimator.fit(X_train,y_train)
                ytrain_star = estimator.predict(X_train)
                ytest_star  = estimator.predict(X_test)
            else:
                estimator.fit(X_train,y_train,K_train)
                ytrain_star = estimator.predict(X_train,K_train)
                ytest_star  = estimator.predict(X_test, K_test)

            MSE_train[ifold,ialpha]  = mean_squared_error(ytrain_star,y_train)
            MSE_test[ifold,ialpha]   = mean_squared_error(ytest_star,y_test)
            W_nonzero[ifold,ialpha]  = SP.sum(estimator.coef_!=0)

        ifold +=1

    t2 = time.time()
    if verbose: print(('finished in %.2f seconds'%(t2-t1)))
    return MSE_train,MSE_test,W_nonzero




class LmmLasso(Lasso):
    """
    Lmm-Lasso classo
    """
    def __init__(self, alpha=1., **lasso_args):
        """
        Extension to the sklearn's LASSO to model population structure

        alpha: l1-regularization parameter
        """
        super(LmmLasso, self).__init__(alpha=alpha, **lasso_args)
        self.msg = 'lmmlasso'




    def fit(self, X, y, K, standardize=False, verbose=False,**lasso_args):
        """
        fitting the model

        X: SNP data
        y: phenotype data
        K: backgroundcovariance matrix
        standardize: if True, genotypes and phenotypes are standardized
        """

        if y.ndim == 2:
            assert y.shape[1]==1, 'Only one phenotype can be processed at at time.'
            y = y.flatten()
        time_start = time.time()
        [n_s, n_f] = X.shape
        assert X.shape[0] == y.shape[0], 'dimensions do not match'
        assert K.shape[0] == K.shape[1], 'dimensions do not match'
        assert K.shape[0] == X.shape[0], 'dimensions do not match'

        """ standardizing genotypes and phenotypes """
        if standardize:
            X -= X.mean(axis=0)
            X /= X.std(axis=0)
            y -= y.mean(axis=0)
            y /= y.std(axis=0)

        """ training null model """
        vd = VarianceDecomposition(y)
        vd.addRandomEffect(is_noise=True)
        vd.addRandomEffect(K)
        vd.optimize()
        varComps = vd.getVarianceComps()
        delta0   = varComps[0,0]/varComps.sum()
        self.varComps = varComps

        S,U = LA.eigh(K)

        """ rotating data """
        Sdi = 1. / (S + delta0)
        Sdi_sqrt = SP.sqrt(Sdi)
        SUX = SP.dot(U.T, X)
        SUX = SUX * SP.tile(Sdi_sqrt, (n_f, 1)).T
        SUy = SP.dot(U.T, y)
        SUy = Sdi_sqrt * SUy

        """ fitting lasso """
        super(LmmLasso, self).fit(SUX, SUy, **lasso_args)
        yhat = super(LmmLasso, self).predict(X)
        self.w_ridge = LA.solve(K + delta0 * SP.eye(n_s), y - yhat)

        time_end = time.time()
        time_diff = time_end - time_start
        if verbose: print(('... finished in %.2fs'%(time_diff)))

        return self

    def getVarianceComps(self,univariance=False):
        """
        Return the estimated variance components

        Args:
            univariance:   Boolean indicator, if True variance components are normalized to sum up to 1 for each trait
        Returns:
            variance components of all random effects [noise, signal]
        """
        if univariance:
            self.varComps /= self.varComps.mean()

        return self.varComps

    def predict(self, Xstar, Kstar=None):
        """
        predicting the phenotype

        Xstar: SNP data
        Kstar: covariance matrix between test and training samples
        """

        assert self.w_ridge.shape[0]==Kstar.shape[1], 'number of training samples is not consistent.'
        assert self.coef_.shape[0]==Xstar.shape[1],   'number of SNPs is not consistent.'
        assert Xstar.shape[0]==Kstar.shape[0],   'number of test samples is not consistent.'

        fixed_effect = super(LmmLasso, self).predict(Xstar)
        if Kstar is not None:
            return fixed_effect + SP.dot(Kstar, self.w_ridge)
        else:
            return fixed_effect
