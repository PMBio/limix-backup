import numpy as np
import numpy.linalg as la
import pandas as pd
from util import *
import time, os
import ipdb
from pysnptools.snpreader import Bed

import logging
import pysnptools.standardizer
import pysnptools
import pysnptools.util
import pysnptools.util.pheno
import time
import kron_gwas
from gwas import GWAS, create_dir
import glob

from limix.core.covar import FreeFormCov
from limix.core.mean import MeanKronSum
from limix.core.gp import GP2KronSum
from limix.core.gp import GP
from limix.utils.preprocess import covar_rescale
from limix.utils.check_grad import mcheck_grad


class MultivariateGWAS(GWAS):
	def __init__(self, snps_test, phenotype, K=None, snps_K=None, covariates=None, 
				 h2=None, interact_with_snp=None, nGridH2=10, standardizer=None, add_bias=True,
				 normalize_K=True, blocksize=10000, A_snps=None, R2=None, C1=None, C2=None):
		"""
		generate a multivariate GWAS object

		Args:
			snps_test:	pysnptools.snpreader object containing the SNPs to be tested
			phenotype:	
			K:			kinship matrix ([N x N] ndarray, optional)
			snps_K:		pysnptools.snpreader object containing the SNPs to be used in the kinship matrix
			covariates:	
			h2:			heritability can be set a priori.
						If not set, the code will perform maximum likelihood estimation (optional)
			interact_with_snp:	
			nGridH2:	number of grid points for h2 optimization
			standardizer:	pysnptools.standardizer to be applied on SNPs (default Unit)
			add_bias:	add a bias term to the covariates? (default True)
			normalize_K:	normalize the kinship matrix to have trace=N
			blocksize:	number of SNPs to load to memory at once when computing the kinship matrix (default 10000)
			A_snps:		The design matrix for SNP testing (default np.eye(N), corresponding to an independent effect test)
			R2:			The noise row covariance matrix (default np.eye(N))
			C1:			The kinship trait x trait covariance matrix (default np.eye(P))
			C2:			The noise trait x trait covariance matrix (default np.eye(P))
		"""
		if standardizer is None:
			standardizer = pysnptools.standardizer.Unit()
		self.standardizer = standardizer
		self.phenotype = GWAS.check_pheno_format(phenotype=phenotype)
		self.covariates = GWAS.check_pheno_format(phenotype=covariates)
		self.snps_test = snps_test
		self.snps_K = snps_K
		self.K = K
		if (self.K is None) and (self.snps_K is None):
			self.K = np.eye(self.sample_count)
		elif self.K is None:
			self.K = snps_K.kernel(standardizer=standardizer, blocksize=blocksize)
			if normalize_K:
				self.K /= self.K.diagonal().mean()
		elif snps_K is None:
			if normalize_K:
				self.K = self.K / self.K.diagonal().mean()
		else:
			raise NotImplementedError("either K or snps_K has to be None")
		if add_bias and (self.covariates is None):
			bias = pd.Series(np.ones((self.sample_count)), index=self.covariates.index)
			self.covariates = pd.DataFrame(data=bias, columns=["bias"])
		elif add_bias:
			bias = pd.Series(np.ones((self.covariates.shape[0])), index=self.covariates.index)
			self.covariates['bias'] = bias
		elif (add_bias==False) and self.covariates is None:
			raise NotImplementedError("currently a model with neither bias term, nor covariates is not supported.")

		if R2 is None:
			self.R2 = np.eye(self.sample_count)
		else:
			self.R2 = R2

		if C1 is None:
			self.C1 = np.eye(self.phenotype.shape[1])
		else:
			self.C1 = C1

		if C2 is None:
			self.C2 = np.eye(self.phenotype.shape[1])
		else:
			self.C2 = C2

		self.A_snps = A_snps
		self.interact_with_snp = interact_with_snp
		self.nGridH2 = nGridH2
		self.lmm = kron_gwas.KroneckerGWAS(Y=self.phenotype.values, R1=self.K, C1=self.C1, R2=self.R2, C2=self.C2, X=self.covariates.values, A=[None], h2=0.5, reml=True)
		if h2 is None:
			self.lmm.find_h2()
		else:
			self.lmm.h2 = h2

	def compute_association(self, blocksize=10000, temp_dir=None):
		"""
		Test all SNP associations

		Args:
			blocksize:	number of SNPs to load in memory at once for computing associations (default: 10000)
			temp_dir:	when specified the results are not kept in memory,
						but rather stored in tem_pdir. This is useful when results get very big. (optional)

		Returns:
			result:		when no temp_dir is specified a pandas.DataFrame containing the GWAS results is returned.
						when temp_dir is specified None is returned.
		"""
		result = None

		if temp_dir is not None:
			if (not os.path.exists(temp_dir)) or (not os.path.isdir(temp_dir)):
				os.makedirs(temp_dir)
		self._generate_intervals(blocksize=blocksize, total_snps=self.snps_test.sid.shape[0])
		for stop in range(1,len(self.intervals)):
			res = self.snps_test_block(block_start=self.intervals[stop-1],block_end=self.intervals[stop])
			if temp_dir is not None:
				self._save_results_block(dir_name=temp_dir, res=res, intervals=self.intervals, idx_interval=stop)
			else:
				if result is None:
					result = res
				else:
					result = pd.concat([result,res])
		return result

	def _save_results_block(self, dir_name, res, intervals=None, idx_interval=None):
		"""
		internal function to save GWAS results in csv format to a temporary directory

		Args:
			dir_name:		name of the temp directory to store results
			res:			results DataFrame to store
			intervals:		total number of intervals to do zero-padding of file names
							(optional for the case when multiple intervals are stored)
			idx_interval:	index of the current interval
		"""	
		mydir = dir_name + "/"
		create_dir(mydir)
		if intervals is not None:
			fill = int(np.floor(np.log10(len(intervals))))+1
		else:
			fill = 0
		if idx_interval is None:
			stop = 1
		else:
			stop = idx_interval
		#block_format = '%0*d' % (fill, stop)
		myfile = "%sblock_%0*d.csv" % (mydir, fill, stop)
		res.to_csv(myfile)

	@staticmethod
	def load_results(dir_name):
		"""
		loads previously generated results from a directory and returns as a single DataFrame

		Args:
			dir_name:		name of the temp directory to store results
		
		Returns:
			pandas.DataFrame of results
		"""
		result = None
		mydir = dir_name + "/"
		myfile_pattern = "%sblock_*.csv" % (mydir)
		files = glob.glob(myfile_pattern)
		for filename in files:
			res = pd.read_csv(filename)
			if result is None:
				result = res
			else:
				result = pd.concat((result,res),0)
		return result

	def snps_test_block(self, block_start, block_end):
		"""
		perform association testing on test_snps[:,block_start:block_end]

		Args:
			block_start:	index of the beginning of block (0-based inclusive)
			block_end:		index of the end of the block (0-based exclusive)

		Returns:
			pandas.DataFrame of GWAS results for the block
		"""
		snps = self.snps_test[:,block_start:block_end].read().standardize(self.standardizer)
		lrts, p_values = self.lmm.run_gwas(snps=snps.val, A_snps=self.A_snps)
		result = self._format_gwas_results(lrts=lrts, p_values=p_values, snps=snps, h2=self.lmm.h2)
		return result

	def _format_gwas_results(self, lrts, p_values, snps, h2):
		"""
		internal function to put GWAS result values into a nice pandas.DataFrame table format
		"""
		items = [
			('SNP', snps.sid),
			('Chr', snps.pos[:,0]), 
			('GenDist', snps.pos[:,1]),
			('ChrPos', snps.pos[:,2]), 
			('PValue', p_values),
			('lrt', lrts),
			('Nullh2', np.zeros((snps.sid_count)) + h2)
		]
		return pd.DataFrame.from_items(items)

if __name__ == "__main__":

	bed_fn = "../../test/data/plinkdata/toydata"
	pheno_fn = bed_fn + ".phe6"#"../../test/data/plinkdata/toydata.phe"
	covariate_fn = 	bed_fn + ".phe"

	
	blocksize = 20000
	snp_reader = Bed(bed_fn)#[:,0:50000]
	#snp_reader.run_once()
	if 1:
		standardizer = pysnptools.standardizer.Unit()
	else:
		standardizer = pysnptools.standardizer.Beta(1,2)
	pheno = pysnptools.util.pheno.loadPhen(filename=pheno_fn,   missing ='-9')
	pheno = GWAS._pheno_fixup(pheno)
	covariates = GWAS._pheno_fixup(covariate_fn, iid_source_if_none=pheno)
	print "intersecting data"
	t00 = time.time()
	snp_intersect, pheno_intersect, covariates_intersect = pysnptools.util.intersect_apply([snp_reader, pheno, covariates], sort_by_dataset=True)
	
	pheno_df = GWAS.check_pheno_format(pheno_intersect)
	covariates_df = GWAS.check_pheno_format(covariates_intersect)

	t1 = time.time()
	print "done intersecting after %.4fs" % (t1-t00)

	print "building kernel"
	t0 = time.time()
	if 0: 
		#snp_data = snp_intersect.read().standardize()
		snp_data = snp_intersect.read().standardize(standardizer)

		G = snp_data.val

		K = G.dot(G.T)
		K/=K.diagonal().mean()
	else:
		K = snp_intersect.kernel(standardizer=standardizer,blocksize=blocksize)
		K /= K.diagonal().mean()
	t1 = time.time()
	print "done building kernel after %.4fs" % (t1-t0)	

	if 0:
		print "computing Eigenvalue decomposition of K"
		t0 = time.time()
		S,U = la.eigh(K)
		t1 = time.time()
		print "done computing eigenvalue decomposition of kernel after %.4fs" % (t1-t0)	
	if 1:
		# define phenotype
		N = pheno_df.shape[0]
		P = pheno_df.shape[1]
		Y = pheno_df.values
		# define fixed effects
		F = []; A = []
		X_cov = np.concatenate((np.ones((N,1)),covariates_df.values),1)
		F.append(X_cov)
		A.append(np.eye(P))
		# define row covariance
		R  = K
		# define col covariances
		Cg = FreeFormCov(P)
		Cn = FreeFormCov(P)
		Cg.setCovariance(0.5 * np.eye(P))
		Cn.setCovariance(0.5 * np.cov(Y.T))
		# define gp
		gp = GP2KronSum(Y=Y, F=F, A=A, Cg=Cg, Cn=Cn, R=R)
		gp.optimize()

		C1 = gp.covar.Cg.K()
		C2 = gp.covar.Cn.K()
	if 1:
		print "running GWAS"
		t0 = time.time()
		mygwas = MultivariateGWAS(K=K, snps_K=None, snps_test=snp_intersect, phenotype=pheno_df, covariates=covariates_df, h2=None, interact_with_snp=None, nGridH2=10, standardizer=standardizer,C1=C1,C2=C2)
		if 1:
			result = mygwas.compute_association(blocksize=blocksize, temp_dir=None)#'./temp_dir_testdata/')
		else:
			result_block = mygwas.snps_test_block(block_start=0, block_end=blocksize)
			mygwas._save_results_block("./res_check/", result_block)
		t1 = time.time()
		print "done running GWAS after %.4fs" % (t1-t0)
		print "total: %.4fs" % (t1-t00)



