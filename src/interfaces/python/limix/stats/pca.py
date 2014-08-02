# Copyright(c) 2014, The LIMIX developers (Christoph Lippert, Paolo Francesco Casale, Oliver Stegle)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
PCA related utility function
"""
import numpy as np 
import pdb
import numpy.linalg as linalg

def PCA(Y, components):
	"""
	run PCA, retrieving the first (components) principle components
	return [s0, eig, w0]
	s0: factors
	w0: weights
	"""

	N,D = Y.shape
	sv = linalg.svd(Y, full_matrices=0);
	[s0, w0] = [sv[0][:, 0:components], np.dot(np.diag(sv[1]), sv[2]).T[:, 0:components]]
	v = s0.std(axis=0)
	s0 /= v;
	w0 *= v;
	return [s0, w0]

	if N>D:
		sv = linalg.svd(Y, full_matrices=0);
		[s0, w0] = [sv[0][:, 0:components], np.dot(np.diag(sv[1]), sv[2]).T[:, 0:components]]
		v = s0.std(axis=0)
		s0 /= v;
		w0 *= v;
		return [s0, w0]
	else:
		K=np.cov(Y)
		sv = linalg.eigh(K)
		std_var = np.sqrt(sv[0])
		pc = sv[1]*std_var[np.newaxis(),0]
		#import ipdb
		#ipdb.set_trace()
		return [pc,std_var]

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
    S,U = linalg.eigh(covY+1e-6*sp.eye(covY.shape[0]))
    S = S[::-1]
    rv = np.array([S[0:i].sum() for i in range(1,S.shape[0])])
    rv/= S.sum()
    return rv

if __name__ == '__main__':
	Y = np.random.randn(10,20)
	components = 5
	#import pca
	pca = PCA(Y, components)
	pass
