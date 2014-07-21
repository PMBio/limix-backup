# Copyright(c) 2014, The LIMIX developers (Christoph Lippert, Paolo Francesco Casale, Oliver Stegle)
# All rights reserved.
#
# LIMIX is provided under a 2-clause BSD license.
# See license.txt for the complete license.

"""
PCA related utility function
"""
import scipy as sp 
import pdb
import scipy.linalg as linalg

def PCA(Y, components):
    """run PCA, retrieving the first (components) principle components
    return [s0, eig, w0]
    s0: factors
    w0: weights
    """
    sv = linalg.svd(Y, full_matrices=0);
    [s0, w0] = [sv[0][:, 0:components], sp.dot(sp.diag(sv[1]), sv[2]).T[:, 0:components]]
    v = s0.std(axis=0)
    s0 /= v;
    w0 *= v;
    return [s0, w0]

def PC_varExplained(Y,standardized=True):
    """
    Run PCA and calculate the cumulative fraction of variance
    Args:
        Y: phenotype values
        standardize: if True, phenotypes are standardized
    Returns:
        var: cumulative distribution of variance explained
    """
    # figuring out the number of latent factors
    if standardized:
        Y-=Y.mean(0)
        Y/=Y.std(0)
    covY = sp.cov(Y)
    S,U = LA.eigh(covY+1e-6*sp.eye(covY.shape[0]))
    S = S[::-1]
    rv = sp.array([S[0:i].sum() for i in range(1,S.shape[0])])
    rv/= S.sum()
    return rv