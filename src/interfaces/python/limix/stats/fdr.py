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
import scipy as sp
import sys, pickle, pdb
import scipy.stats as st
import scipy.interpolate
import logging as LG


def qvalues(pv, m = None, return_pi0 = False, lowmem = False, pi0 = None, fix_lambda = None):

    original_shape = pv.shape

    assert(pv.min() >= 0 and pv.max() <= 1)

    pv = pv.ravel() # flattens the array in place, more efficient than flatten() 

    if m == None:
	m = float(len(pv))
    else:
	# the user has supplied an m, let's use it
	m *= 1.0

    # if the number of hypotheses is small, just set pi0 to 1
    if len(pv) < 100:
	pi0 = 1.0
    elif pi0 != None:
	pi0 = pi0
    else:
	# evaluate pi0 for different lambdas
	pi0 = []
	lam = sp.arange(0, 0.90, 0.01)
	counts = sp.array([(pv > i).sum() for i in sp.arange(0, 0.9, 0.01)])
	
	if fix_lambda != None:
	    interv_count = (pv > fix_lambda - 0.01).sum()
	    uniform_sim = sp.array([(pv > fix_lambda-0.01).sum()*(i+1) for i in sp.arange(0, len(sp.arange(0, 0.90, 0.01)))][::-1])
	    counts += uniform_sim
	    
	for l in range(len(lam)):
	    pi0.append(counts[l]/(m*(1-lam[l])))

	pi0 = sp.array(pi0)

	# fit natural cubic spline
	tck = sp.interpolate.splrep(lam, pi0, k = 3)
	pi0 = sp.interpolate.splev(lam[-1], tck)
	if pi0 > 1:
	    LG.warning("got pi0 > 1 (%.3f) while estimating qvalues, setting it to 1" % pi0)
	    pi0 = 1.0

	assert(pi0 >= 0 and pi0 <= 1), "%f" % pi0


    if lowmem:
	# low memory version, only uses 1 pv and 1 qv matrices
	qv = sp.zeros((len(pv),))
	last_pv = pv.argmax()
	qv[last_pv] = (pi0*pv[last_pv]*m)/float(m)
	pv[last_pv] = -sp.inf
	prev_qv = last_pv
	for i in xrange(int(len(pv))-2, -1, -1):
	    cur_max = pv.argmax()
	    qv_i = (pi0*m*pv[cur_max]/float(i+1))
	    pv[cur_max] = -sp.inf
	    qv_i1 = prev_qv
	    qv[cur_max] = min(qv_i, qv_i1)
	    prev_qv = qv[cur_max]

    else:
	p_ordered = sp.argsort(pv)    
	pv = pv[p_ordered]
	# estimate qvalues
# 	qv = pi0*m*pv/(sp.arange(len(pv))+1.0)
	
# 	for i in xrange(int(len(qv))-2, 0, -1):
# 	    qv[i] = min([qv[i], qv[i+1]])


	qv = pi0 * m/len(pv) * pv
	qv[-1] = min(qv[-1],1.0)

	for i in xrange(len(pv)-2, -1, -1):
	    qv[i] = min(pi0*m*pv[i]/(i+1.0), qv[i+1])



	# reorder qvalues
	qv_temp = qv.copy()
	qv = sp.zeros_like(qv)
	qv[p_ordered] = qv_temp



    # reshape qvalues
    qv = qv.reshape(original_shape)

    if return_pi0:
	return qv, pi0
    else:
	return qv


def estimate_lambda(pv):
    """estimate lambda form a set of PV"""
    LOD2 = sp.median(st.chi2.isf(pv,1))
    L = (LOD2/0.456)
    return (L)

def LOD2PV(lods):
    PV = (st.chi2.sf(2*lods, 1))
    return PV
        