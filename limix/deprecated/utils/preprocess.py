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
"""preprocessing functions"""

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
        print(('Kinship scaled by: %0.4f' % scalar))
    K = K * scalar
    return K


def standardize(Y,in_place=False):
    """
    standardize Y in a way that is robust to missing values
    in_place: create a copy or carry out inplace opreations?
    """
    if in_place:
        YY = Y
    else:
        YY = Y.copy()
    for i in range(YY.shape[1]):
        Iok = ~SP.isnan(YY[:,i])
        Ym = YY[Iok,i].mean()
        YY[:,i]-=Ym
        Ys = YY[Iok,i].std()
        YY[:,i]/=Ys
    return YY
    

def rankStandardizeNormal(X):
	"""
	Gaussianize X: [samples x phenotypes]
	- each phentoype is converted to ranks and transformed back to normal using the inverse CDF
	"""
	Is = X.argsort(axis=0)
	RV = SP.zeros_like(X)
	rank = SP.zeros_like(X)
	for i in range(X.shape[1]):
		x =  X[:,i]
		i_nan = SP.isnan(x)
		if 0:
			Is = x.argsort()
			rank = SP.zeros_like(x)
			rank[Is] = SP.arange(X.shape[0])
			#add one to ensure nothing = 0
			rank +=1
		else:
			rank = st.rankdata(x[~i_nan])
		#devide by (N+1) which yields uniform [0,1]
		rank /= ((~i_nan).sum()+1)
		#apply inverse gaussian cdf
		RV[~i_nan,i] = SP.sqrt(2) * special.erfinv(2*rank-1)
		RV[i_nan,i] = x[i_nan]
	return RV


def boxcox(X):
	"""
    Gaussianize X using the Box-Cox transformation: [samples x phenotypes]

    - each phentoype is brought to a positive schale, by first subtracting the minimum value and adding 1.
    - Then each phenotype transformed by the boxcox transformation
	"""
	X_transformed = SP.zeros_like(X)
	maxlog = SP.zeros(X.shape[1])
	for i in range(X.shape[1]):
		i_nan = SP.isnan(X[:,i])
		values = X[~i_nan,i]
		X_transformed[i_nan,i] = X[i_nan,i]
		X_transformed[~i_nan,i], maxlog[i] = st.boxcox(values-values.min()+1.0)
	return X_transformed, maxlog


