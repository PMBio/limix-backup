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

import numpy as np
import scipy.stats as st

def power(maf=0.5,beta=0.1, N=100, cutoff=5e-8):
	"""
	estimate power for a given allele frequency, effect size beta and sample size N

	Assumption:
	
	z-score = beta_ML distributed as p(0) = N(0,1.0(maf*(1-maf)*N))) under the null hypothesis
	the actual beta_ML is distributed as p(alt) = N( beta , 1.0/(maf*(1-maf)N) )
	

	
	Arguments:
		maf:	minor allele frequency of the SNP
		beta:	effect size of the SNP
		N:		sample size (number of individuals)
	Returns:
		power:	probability to detect a SNP in that study with the given parameters
	"""
	
	"""
	std(snp)=sqrt(2.0*maf*(1-maf)) 
	power = \int 

	beta_ML = (snp^T*snp)^{-1}*snp^T*Y = cov(snp,Y)/var(snp)   
	E[beta_ML]	= (snp^T*snp)^{-1}*snp^T*E[Y]  
				= (snp^T*snp)^{-1}*snp^T*snp * beta
				= beta
	Var[beta_ML]= (snp^T*snp)^{-1}*(snp^T*snp)*(snp^T*snp)^{-1}
				= (snp^T*snp)^{-1}
				= 1/N * var(snp)
				= 1/N * maf*(1-maf)
	"""
	assert maf>=0.0 and maf<=0.5, "maf needs to be between 0.0 and 0.5, got %f" % maf
	if beta<0.0:
		beta=-beta
	std_beta = 1.0/np.sqrt(N*(2.0 * maf*(1.0-maf)))
	non_centrality = beta
	beta_samples = np.random.normal(loc=non_centrality, scale=std_beta)
	n_grid = 100000
	beta_in = np.arange(0.5/(n_grid+1.0),(n_grid-0.5)/(n_grid+1.0),1.0/(n_grid+1.0)) 
	beta_theoretical = ((st.norm.isf(beta_in)* std_beta) + non_centrality)
	pvals = st.chi2.sf( (beta_theoretical/std_beta)*(beta_theoretical/std_beta) ,1.0) 
	
	power = (pvals<cutoff).mean()
	return power, pvals

