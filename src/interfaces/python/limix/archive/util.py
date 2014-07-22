"""@package util
misc. utility functions used in limix modules and demos
"""

import numpy as np
import scipy as sp
import scipy as SP
import pdb, sys, pickle
import matplotlib.pylab as plt
import scipy.stats as st
import scipy.interpolate



def mean_impute(X, imissX=None, maxval=2.0):
    if imissX is None:
        imissX = np.isnan(X)
    
    n_i,n_s=X.shape
    if imissX is None:
        n_obs_SNP=np.ones(X.shape)
    else:    
        i_nonan=(~imissX)
        n_obs_SNP=i_nonan.sum(0)
        X[imissX]=0.0
    snp_sum=(X).sum(0)
    one_over_sqrt_pi=(1.0+snp_sum)/(2.0+maxval*n_obs_SNP)
    one_over_sqrt_pi=1./np.sqrt(one_over_sqrt_pi*(1.-one_over_sqrt_pi))
    snp_mean=(snp_sum*1.0)/(n_obs_SNP)

    X_ret=X-snp_mean
    X_ret*=one_over_sqrt_pi
    if imissX is not None:
        X_ret[imissX]=0.0
    return X_ret

def getPosNew(data):
    """
    get Fixed position
    """
    pos = data.geno['col_header']['pos'][:]
    chrom= data.geno['col_header']['chrom'][:]
    n_chroms = chrom.max()
    pos_new = []
    for chrom_i in range(1,n_chroms+1):
        I = chrom==chrom_i
        _pos = pos[I]
        for i in range(1,_pos.shape[0]):
            if not _pos[i]>_pos[i-1]:
                _pos[i:]=_pos[i:]+_pos[i-1]
        pos_new.append(_pos)
    pos_new = SP.concatenate(pos_new)
    return pos_new

def getCumPos(data):
    """
    getCumulativePosition
    """
    pos = getPosNew(data)
    chrom= data.geno['col_header']['chrom'][:]
    n_chroms = int(chrom.max())
    x = 0
    for chrom_i in range(1,n_chroms+1):
        I = chrom==chrom_i
        pos[I]+=x
        x=pos[I].max()
    return pos

def getChromBounds(data):
    """
    getChromBounds
    """
    chrom= data.geno['col_header']['chrom'][:]
    posCum = getCumPos(data)
    n_chroms = int(chrom.max())
    chrom_bounds = []
    for chrom_i in range(2,n_chroms+1):
        I1 = chrom==chrom_i
        I0 = chrom==chrom_i-1
        _cb = 0.5*(posCum[I0].max()+posCum[I1].min())
        chrom_bounds.append(_cb)
    return chrom_bounds