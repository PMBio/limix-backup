"""preprocessing functions for ..."""

import scipy as SP
import scipy.special as special


def scale_K(K, verbose=False):
    """scale covariance K such that it explains unit variance"""
    c = SP.sum((SP.eye(len(K)) - (1.0 / len(K)) * SP.ones(K.shape)) * SP.array(K))
    scalar = (len(K) - 1) / c
    if verbose:
        print 'Kinship scaled by: %0.4f' % scalar
    K = scalar * K
    return K


def rankStandardizeNormal(X):
    """
    standardize X: [samples x phenotypes]
    - each phentoype is converted to ranks and transformed back to normal using the inverse CDF
    """
    rank = (1+SP.array(X.argsort(axis=0),dtype='float'))
    rank /= (X.shape[0]+1)
    #apply inverse gaussian cdf
    RV = SP.sqrt(2) * special.erfinv(2*rank-1)
    return RV
