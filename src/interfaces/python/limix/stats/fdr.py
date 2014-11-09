"""@package fdr
FDR estimation using Benjamini Hochberg and Stories method
"""
import scipy as SP
import numpy as NP
import scipy.interpolate as INTER
import scipy.stats as STATS

import sys
import pdb
import logging as LG


def lfdr(pv,pi0_est,pi0_set=None,eps=1e-8,trunc=True,monotone=True,debug=False,pi0_min=0.05,pi0_max=0.95):
    """
    estimate local false discovery rate

    input:
    pv        : p-values
    p0_est    : prior of being null (estimated)
    p0_set    : prior of being null (desired)
    eps       : p-value is squeezed into the inverval [eps, 1-eps]
    trunc     : truncate lfdr
    montone   : ?
    """
    if pi0_est<pi0_min: pi0_est=pi0_min
    if pi0_est>pi0_max: pi0_est=pi0_max
        
    original_shape = pv.shape
    assert(pv.min() >= 0 and pv.max() <= 1), 'p-values are not in a valid range.'
    pv = pv.ravel() 
    n  = pv.shape[0]


    x   = SP.log((pv+eps)/(1-pv+eps))
    f_kde = STATS.gaussian_kde(x)
    y_kde = f_kde.evaluate(x)
    dx   = SP.exp(x)/(1+SP.exp(x))**2
    lfdr  = pi0_est*dx/y_kde
    info = {'x':x,'y_kde':y_kde,'lfdr_raw':lfdr,'dx':dx}

    if pi0_set!=None:
        lfdr = pi0_set *1 /((1-pi0_set)/(1-pi0_est)*(y_kde/dx)+pi0_set-(1-pi0_set)*pi0_est/(1-pi0_est))
 
        
    if trunc:
        """
        local false discovery rate must be between one and zero
        """
        if SP.any(lfdr>1):
            LG.warning("#{lfdr>1}=%d, setting it to one." % SP.sum(lfdr>1))
            lfdr[lfdr>1] = 1
        if SP.any(lfdr<0):
            LG.warning("#{lfdr<0}=%d, setting it to one." % SP.sum(lfdr<0))
            lfdr[lfdr<0] = 0

    if monotone:
        """
        the smaller the p-value, the smaller the posterior probabibility of being null
        """
        order = SP.argsort(pv)
        lfdr = lfdr[order]
        for i in range(1,n):
            if lfdr[i]<lfdr[i-1]: lfdr[i] = lfdr[i-1]
        rank = SP.argsort(order)
        lfdr = lfdr[rank]

    return lfdr, info
    
    
def qvalues(pv, lam=None, pi0 = None):
    """
    computing q-values

    Input:
       pv:  p-values
       pi:  prior probability of being null (default: None)
       lam: threshold array used for estimating pi if not provided
    """
    original_shape = pv.shape
    assert(pv.min() >= 0 and pv.max() <= 1), 'p-values are not in a valid range.'
    pv = pv.ravel() 
    n  = pv.shape[0]
    info = {}
    
    if pi0==None:
        """
        setting lambda
        """
        if lam==None:
            lam = SP.arange(0, 0.90, 0.05)
        if len(lam)>1: assert len(lam)>4, 'if length of lambda greater than 1, you need at least 4 values.'
        assert min(lam)>=0 or max(lam)<=1, 'lambda must be in [0,1)'
        info['lam'] = lam
        
        if len(lam)==1:
            """
            lambda is fixed
            """
            pi0 = SP.mean(pv >= lam)/(1-lam)
            pi0 = min(pi0,1)
            
        else:
            """
            evaluating for different lambdas
            """
            pi0_arr = SP.zeros(len(lam))
            for i in range(len(lam)):
                pi0_arr[i] = SP.sum(pv>lam[i])/(n*(1-lam[i]))

            """
            smoothing
            """
            f_spline = INTER.UnivariateSpline(lam,pi0_arr,k=3,s=None)
            pi0 = f_spline(lam[-1])
            if pi0 > 1:
                LG.warning("got pi0 > 1 (%.3f) while estimating qvalues, setting it to 1" % pi0)
                pi0 = 1.0
            info['pi0_arr'] = pi0_arr
            info['pi0_est'] = f_spline(lam)
            assert(pi0 <= 1), "estimated pi0 is greater than 1"


            #pdb.set_trace()

        info['pi0'] = pi0
        
        """
        computing q-values
        """
        pv_ordered = SP.argsort(pv)
	pv = pv[pv_ordered]
	qv = pi0 * pv
	qv[-1] = min(qv[-1],1.0)
	for i in xrange(len(pv)-2, -1, -1):
	    qv[i] = min(pi0*n*pv[i]/(i+1.0), qv[i+1])
	qv_temp = qv.copy()
	qv = SP.zeros_like(qv)
	qv[pv_ordered] = qv_temp
        qv = qv.reshape(original_shape)
        
        return qv,info


def pvalues(stats,stats0,pooled=True):
    """
    compute pvalues out of observed and permuted test statistics. if pooled is true, pool permuted
    test statiscs.
    """

    if pooled:
        n_stats  = len(stats)
        stats  = stats.ravel()
        stats0 = stats0.ravel()
        n_stats0 = len(stats0)
        B = n_stats0/n_stats

        indObs = SP.zeros(n_stats + n_stats0,dtype=bool)
        indObs[:n_stats] = True
        v     = SP.hstack([stats,stats0])
        order = SP.argsort(-v)
        indObs= indObs[order]

        u = SP.arange(n_stats + n_stats0)
        w = SP.arange(n_stats)
        pv = (1.*(u[indObs] -w))/n_stats0

        order = SP.argsort(-stats)
        rank  = SP.argsort(order)
        pv    = pv[rank]

        pv_min = 1./n_stats0
        pv[pv<pv_min] = pv_min

        return pv
    
    else:
        B = stats0.shape[1]

        if stats.ndim==1:
            stats = stats[:,SP.newaxis]
        pv = (stats0 - SP.repeat(stats,B,axis=1)) >= 0
        pv = pv.mean(1)
        pv_min = 1./B
        pv[pv<pv_min] = pv_min
        return pv
    
