"""preprocessing functions for ..."""

import scipy as SP
import scipy.special as special
import scipy.stats as st
import pdb


def variance_K(K, verbose=False):
    """estimate the variance explained by K"""
    c = SP.sum((SP.eye(len(K)) - (1.0 / len(K)) * SP.ones(K.shape)) * SP.array(K))
    scalar = (len(K) - 1) / c
    return 1.0/scalar


def scale_K(K, verbose=False,trace_method=True):
    """scale covariance K such that it explains unit variance
    trace_method: standardize to unit trace (deafault: True)
    """
    if trace_method:
        scalar=1.0/(K.diagonal().mean())
    else:
        c = SP.sum((SP.eye(len(K)) - (1.0 / len(K)) * SP.ones(K.shape)) * SP.array(K))
        scalar = (len(K) - 1) / c
    if verbose:
        print 'Kinship scaled by: %0.4f' % scalar
    K = K * scalar
    return K


def standardize(Y,in_place=False):
    """
    standardize Y in a way that is robust to missing values
    in_plcase: create a copy or carry out inplace opreations?
    """
    if in_place:
        YY = Y
    else:
        YY = Y.copy()
    for i in xrange(YY.shape[1]):
        Iok = ~SP.isnan(YY[:,i])
        Ym = YY[Iok,i].mean()
        YY[:,i]-=Ym
        Ys = YY[Iok,i].std()
        YY[:,i]/=Ys
    return YY
    

def rankStandardizeNormal(X):
    """
    standardize X: [samples x phenotypes]
    - each phentoype is converted to ranks and transformed back to normal using the inverse CDF
    """
    Is = X.argsort(axis=0)
    RV = SP.zeros_like(X)
    rank = SP.zeros_like(X)
    for i in xrange(X.shape[1]):
        x =  X[:,i]
        if 0:
            Is = x.argsort()
            rank = SP.zeros_like(x)
            rank[Is] = SP.arange(X.shape[0])
            #add one to ensure nothing = 0
            rank +=1
        else:
            rank = st.rankdata(x)
        #devide by (N+1) which yields uniform [0,1]
        rank /= (X.shape[0]+1)
        #apply inverse gaussian cdf
        RV[:,i] = SP.sqrt(2) * special.erfinv(2*rank-1)
    return RV
