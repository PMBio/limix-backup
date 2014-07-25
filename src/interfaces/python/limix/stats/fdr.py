# Copyright(c) 2014, The LIMIX developers (Christoph Lippert, Paolo Francesco Casale, Oliver Stegle)
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

"""@package fdr
FDR estimation using Benjamini Hochberg and Stories method
"""

import numpy as np
import sys, pickle, pdb
import scipy.stats as st
import scipy.interpolate as inter
import logging as lg



def lfdr(pv,pi0,eps=1e-8,trunc=True,monotone=True,bw_method=1.5):
	"""
	estimate local false discovery rate using logistic regression

	Input:
		pv        : p-values
		p0        : prior of being null
		eps       : p-value is squeezed into the inverval [eps, 1-eps]
		trunc     : truncate lfdr
		monotone  : ?
		bw_method : used to calculate the estimator bandwidth (see
					http://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html#scipy.stats.gaussian_kde
					for more information)

	Returns:
		lfdr:	estimated local FDRs
		info:	{'x':x,'y_kde':y_kde,'lfdr_raw':lfdr,'dx':x}
	"""
	original_shape = pv.shape
	assert(pv.min() >= 0 and pv.max() <= 1), 'p-values are not in a valid range.'
	pv = pv.ravel() 
	n  = pv.shape[0]

	x   = np.log((pv+eps)/(1-pv+eps))
	f_kde = st.gaussian_kde(x,bw_method=bw_method)
	y_kde = f_kde.evaluate(x)
	f_spline = inter.UnivariateSpline(x,y_kde,k=3)
	y_spline = f_spline(x)
	dx   = np.exp(x)/(1+np.exp(x))**2
	#lfdr = pi0*dx/y_spline
	lfdr  = pi0*dx/y_kde
	#info = {'x':x,'y_kde':y_kde,'y_spline':y_spline,'lfdr_raw':lfdr,'dx':x}
	info = {'x':x,'y_kde':y_kde,'lfdr_raw':lfdr,'dx':x}

	if trunc:
		"""
		local false discovery rate must be between one and zero
		"""
		if np.any(lfdr>1):
			lg.warning("#{lfdr>1}=%d, setting it to one." % np.sum(lfdr>1))
			lfdr[lfdr>1] = 1
		if np.any(lfdr<0):
			lg.warning("#{lfdr<0}=%d, setting it to one." % np.sum(lfdr<0))
			lfdr[lfdr<0] = 0

	if monotone:
		"""
		the smaller the p-value, the smaller the posterior probabibility of being null
		"""
		order = np.argsort(pv)
		lfdr = lfdr[order]
		for i in range(1,n):
			if lfdr[i]<lfdr[i-1]: lfdr[i] = lfdr[i-1]
		rank = np.argsort(order)
		lfdr = lfdr[rank]

	return lfdr, info


def qvalues(pv, lam=None, pi0 = None,cv=True):
	"""
	computing q-values

	Input:
		pv:  p-values
		pi:  prior probability of being null (default: None)
		lam: threshold array used for estimating pi if not provided

	Returns:
		qv:		estimated q-values
		info:	'lam' estimated lambda parameter
				'pi0_arr'
				'pi0_est'	f_spline(lam)
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
			lam = np.arange(0, 0.90, 0.05)
		if len(lam)>1: assert len(lam)>4, 'if length of lambda greater than 1, you need at least 4 values.'
		assert min(lam)>=0 or max(lam)<=1, 'lambda must be in [0,1)'
		info['lam'] = lam

		if len(lam)==1:
			"""
			lambda is fixed
			"""
			pi0 = np.mean(pv >= lam)/(1-lam)
			pi0 = min(pi0,1)

		else:
			"""
			evaluating for different lambdas
			"""
			pi0_arr = np.zeros(len(lam))
			for i in range(len(lam)):
				pi0_arr[i] = np.sum(pv>lam[i])/(n*(1-lam[i]))

			"""
			smoothing
			"""
			if cv:
				smoothing_factor = [1e-3,1e-2,1e-1,1,10]
				MSE = np.zeros(len(smoothing_factor))
				for ismooth in range(len(smoothing_factor)):
					f_spline = inter.UnivariateSpline(lam,pi0_arr,k=3,s=smoothing_factor[ismooth])

					for itest in range(1,len(lam)-1):
						idx      = np.ones(len(lam),dtype=bool)
						idx[itest] = False
						f_spline = inter.UnivariateSpline(lam[idx],pi0_arr[idx],k=3,s=smoothing_factor[ismooth])
						MSE[ismooth] += (pi0_arr[itest] - f_spline(lam[itest]))**2

				idx = np.argmin(MSE)
				s = smoothing_factor[idx]
			else:
				s = None

			f_spline = inter.UnivariateSpline(lam,pi0_arr,k=3,s=s)
			pi0 = f_spline(lam[-1])
			if pi0 > 1:
				lg.warning("got pi0 > 1 (%.3f) while estimating qvalues, setting it to 1" % pi0)
				pi0 = 1.0
			info['pi0_arr'] = pi0_arr
			info['pi0_est'] = f_spline(lam)
			assert(pi0 <= 1), "estimated pi0 is greater than 1"

		info['pi0'] = pi0

		"""
		computing q-values
		"""
		pv_ordered = np.argsort(pv)
		pv = pv[pv_ordered]
		qv = pi0 * pv
		qv[-1] = min(qv[-1],1.0)
		for i in xrange(len(pv)-2, -1, -1):
			qv[i] = min(pi0*n*pv[i]/(i+1.0), qv[i+1])
		qv_temp = qv.copy()
		qv = np.zeros_like(qv)
		qv[pv_ordered] = qv_temp
		qv = qv.reshape(original_shape)

		return qv,info


def pvalues(stats,stats0,pooled=True):
    """
    compute pvalues out of observed and permuted test statistics. if pooled is true, pool permuted
    test statiscs.
	
	Input:
		stats:	unpermuted test statistics
		stats0:	permuted test statistics
		pooled:	(bool) pool the permutations? (default True)
	Returns:
		pv:	P-values
    """

    if pooled:
        n_stats  = len(stats)
        stats  = stats.ravel()
        stats0 = stats0.ravel()
        n_stats0 = len(stats0)
        B = n_stats0/n_stats

        indObs = np.zeros(n_stats + n_stats0,dtype=bool)
        indObs[:n_stats] = True
        v     = np.hstack([stats,stats0])
        order = np.argsort(-v)
        indObs= indObs[order]

        u = np.arange(n_stats + n_stats0)
        w = np.arange(n_stats)
        pv = (1.*(u[indObs] -w))/n_stats0

        order = np.argsort(-stats)
        rank  = np.argsort(order)
        pv    = pv[rank]

        pv_min = 1./n_stats0
        pv[pv<pv_min] = pv_min

        return pv
    
    else:
        B = stats0.shape[1]

        if stats.ndim==1:
            stats = stats[:,np.newaxis]
        pv = (stats0 - np.repeat(stats,B,axis=1)) >= 0
        pv = pv.mean(1)
        pv_min = 1./B
        pv[pv<pv_min] = pv_min
        return pv
   


def estimate_lambda(pv):
    """estimate lambda form a set of PV"""
    LOD2 = np.median(st.chi2.isf(pv,1))
    L = (LOD2/0.456)
    return (L)

def LOD2PV(lods):
	"""
	compute P-values from log likelihood ratios
	PV = (st.chi2.sf(2*lods, 1))

	Input:
		lods: log likelihood ratios
	Returns:
		P-values
	"""
	PV = (st.chi2.sf(2*lods, 1))
	return PV