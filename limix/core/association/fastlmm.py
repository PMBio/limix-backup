# Copyright 2014 Microsoft Corporation

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# original file at https://github.com/MicrosoftGenomics/FaST-LMM

# modified 2015 by Christoph Lippert

import numpy as np
import numpy.linalg as la
import scipy.optimize as opt
import scipy.stats as st
import scipy.special as ss
import pandas as pd
from mingrid import *
from util import *
import time, os
import ipdb
from pysnptools.snpreader import Bed

class LMM(object):
	'''
	linear mixed model with up to two kernels
	N(y | X*beta + covariates*alpha; sigma2(h2*K + (1-h2)*I),
	where
	K = G*G^T

	Relative to lmm.py, lmm_cov handles multiple phenotypes, and regresses out covariates before GWAS testing (which makes it O(#covs^3) faster).
	Currently, h2 for each phenotype is the same, but much of the code is written to allow h2 to be different (see e.g. nlleval_2k).

    This class is intended for a single full-rank kernel (and optionally a low-rank kernel). If two full-rank kernels are at hand, 
    they must first be combined into a single full-rank kernel.

    TODO: Add full support for multiple phenotypes by extending h2 and a2
    TODO: Deal with h2_1 parameterization more eloquently (see comments below)
	'''
	__slots__ = ["linreg","G","Y","X","K","U","S","UX","UY","UUX","UUY","forcefullrank","regressX","numcalls"]

	def __init__(self, forcefullrank=False, X=None, linreg=None, Y=None, G=None, K=None, regressX=True, inplace=False):
		'''

		Args:
		    forcefullrank   : if True, then the code always computes K and runs cubically (default: False)
		     X              : [N x D] np.array of D covariates for N individuals (including bias term).
		                          If None is given, only a bias term is used. (Default: None)
		     linreg         : optionally allows to provide pre-computed Linreg object instead of covariates
		                         (default: None)
		     Y              : [N x P] np.array holding P phenotypes for N individuals
		     G              : [N x k] np.array holding k design vectors for building linear kernel K
		     K              : [N x N] np.array positive semi-definite Kernel matrix
		     regressX       : regress out covariates to speed up computations? (default: True)
		     inplace        : set kernel without copying? (default: False)
		'''
		self.numcalls = 0
		self.setX(X=X, regressX=regressX, linreg=linreg)    #set the covariates (needs to be first)
		self.forcefullrank = forcefullrank
		self.setK(K=K, G=G, inplace=inplace)                 #set the kernel, if available
		self.setY(Y=Y)                      #set the phenotypes

	def setY(self, Y):
		'''
		set the phenotype y.
		
		Args:
		    Y              : [N x P] np.array holding P phenotypes for N individuals
		'''
		self.Y = Y
		self.UY = None
		self.UUY = None

	def setX(self, X=None, regressX=True, linreg=None):
		"""
		set the covariates.
		
		Args:
		     X:           [N x D] np.array of D covariates for N individuals (including bias term).
		                      If None is given, only a bias term is used. (Default: None)
		     regressX:    regress out covariates to speed up computations? (default: True)
		     linreg:      optionally allows to provide pre-computed Linreg object instead of covariates
		                      (default: None)
		"""
		self.X = X
		self.UX = None
		self.UUX = None
		self.linreg = linreg
		self.regressX = regressX
		if self.linreg is None and regressX:
			self.linreg = Linreg(X=self.X)


	def setSU_fromK(self):
		"""
		compute the spectral decomposition from full kernel (full rank computations).
		"""
		N = self.K.shape[0]
		D = self.linreg.D
		ar = np.arange(self.K.shape[0])
		self.K[ar,ar]+=1.0
		K_ = self.linreg.regress(Y=self.K)
		K_ = self.linreg.regress(Y=K_.T)
		[self.S,self.U] = la.eigh(K_)
		self.U = self.U[:,D:N]
		self.S = self.S[D:N] - 1.0


	def setSU_fromG(self):
		"""
		compute the spectral decomposition from design matrix G for linear kernel
		        (allows low rank computations, if G has more rows than columns) 
		"""
		k = self.G.shape[1]
		N = self.G.shape[0]
		if k:
			if ((not self.forcefullrank) and (k < N)):
				#it is faster using the eigen decomposition of G.T*G but this is more
				#accurate
				PxG = self.linreg.regress(Y=self.G)
				try:
					[self.U,self.S,V] = la.svd(PxG,False,True)
					inonzero = self.S > 1E-10
					self.S = self.S[inonzero]
					self.S = self.S * self.S
					self.U = self.U[:,inonzero]
                
				except la.LinAlgError:  # revert to Eigenvalue decomposition
					print "Got SVD exception, trying eigenvalue decomposition of square of G. Note that this is a little bit less accurate"
					[S,V] = la.eigh(PxG.T.dot(PxG))
					inonzero = (S > 1E-10)
					self.S = S[inonzero]
					#self.S*=(N/self.S.sum())
					self.U = self.G.dot(V[:,inonzero] / np.sqrt(self.S))
			else:
				K = self.G.dot(self.G.T)
				self.setK(K=K)
				self.setSU_fromK()
			pass
		else:#rank of kernel = 0 (linear regression case)
			self.S = np.zeros((0))
			self.U = np.zeros_like(self.G)


	def getSU(self):
		"""
		get the spectral decomposition of the kernel matrix. Computes it if needed.
		"""
		if self.U is None or self.S is None:
			if self.K is not None:
				self.setSU_fromK()
			elif self.G is not None:
				self.setSU_fromG()
			else:
				raise Exception("No Kernel is set. Cannot return U and S.") 
		return self.S, self.U

	def rotate(self, A):
		"""
		rotate a matrix A with the eigenvalues of the kernel matrix.
		
		Args:
		           A:     [N x D] np.array
		Returns:
		           U.T.dot(A)
			   A - U.dot(U.T.dot(A))    (if kernel is full rank this is None)
		"""
		S,U = self.getSU()
		N = A.shape[0]
		D = self.linreg.D
		if (S.shape[0] < N - D):#lowrank case
			A = self.linreg.regress(A)
			UA = self.U.T.dot(A)
			UUA = A - U.dot(UA)
		else:
			#A=self.linreg.regress(A)
			UA = U.T.dot(A)
			#A1 = UA=U.T.dot(A)
			#diff = np.absolute(A1-UA).sum()
			#print diff
			#print UA.shape
			UUA = None
		return UA,UUA


	def getUY(self, idx_pheno=None):
		"""
		get the rotated phenotype matrix

		Args:
			idx_pheno	boolean numpy.array of phenotype indices to use
						(default None, returns all)	
		Returns:
			UY		U.T.dot(Y) rotated phenotypes
			UUY		None if kernel is full rank, otherwise Y-U.dot(U.T.dot(Y))
		"""

		if self.UY is None:
			self.UY,self.UUY = self.rotate(A=self.Y)
		if idx_pheno is None:
			return self.UY,self.UUY
		else:
			UY = self.UY[:,idx_pheno]
			UUY = None
			if self.UUY is not None:
				UUY = self.UUY[:,idx_pheno]
			return UY, UUY


	def setK(self, K=None, G=None, inplace=False):
		'''
		set the Kernel K.
		
		Args:
		        K :       [N*N] array, random effects covariance (positive semi-definite)
			G :       [NxS] array of random effects (will be used for linear kernel)
			inplace:  set kernel without copying? (default: False)
		'''
		self.clear_cache()
		if K is not None:
			if inplace:
				self.K = K
			else:
				self.K = K.copy()
		elif G is not None:
			if inplace:
				self.G = G
			else:
				self.G = G.copy()
        

	def clear_cache(self, reset_K=True):
		"""
		delete all cached objects
		
		Args:
		       reset_K: also delete the kernel matrix and the kernel design matrix G? (default: True)
		"""
		self.U = None
		self.S = None
		self.UY = None
		self.UUY = None
		self.UX = None
		self.UUX = None          
		if reset_K:
			self.G = None
			self.K = None




	def findA2_2K(self, nGridA2=10, minA2=0.0, maxA2=1.0, verbose=False, i_up=None, i_G1=None, UW=None, UUW=None, h2=0.5, **kwargs):
		'''
		For a given h2, finds the optimal kernel mixture weight a2 and returns the negative log-likelihood
		
		Find the optimal a2 given h2, such that K=(1.0-a2)*K0+a2*K1. Performs a double loop optimization (could be expensive for large grid-sizes)
		(default maxA2 value is set to 1 as loss of positive definiteness of the final model covariance only depends on h2, not a2)
		
		Allows to provide a second "low-rank" kernel matrix in form of a rotated design matrix W
		second kernel K2 = W.dot(W.T))

		W may hold a design matrix G1 of a second kernel and some columns that are identical to columns of the design matrix of the first kernel to enable subtracting out sub kernels (as for correcting for proximal contamination)

        Args:
		    h2      : "heritabiliy" of the kernel matrix
		    nGridA2 : number of a2-grid points to evaluate the negative log-likelihood at. Number of grid points for Brent search intervals (default: 10)
		    minA2   : minimum value for a2 optimization
		    maxA2   : maximum value for a2 optimization
		    verbose : verbose output? (default: False)
		    i_up    : indices of columns in W corresponding to columns from first kernel that are subtracted of
		    i_G1    : indeces of columns in W corresponding to columns of the design matrix for second kernel G1
		    UW      : U.T.dot(W), where W is [N x S2] np.array holding the design matrix of the second kernel
		    UUW     : W - U.dot(U.T.dot(W))     (provide None if U is full rank)

		Returns:
		    dictionary containing the model parameters at the optimal a2
		'''
		if self.Y.shape[1] > 1:
			print "not implemented"
			raise NotImplementedError("only single pheno case implemented")
        
		self.numcalls = 0
		resmin = [None]
		def f(x,resmin=resmin, **kwargs):
			self.numcalls+=1
			t0 = time.time()
			h2_1 = (1.0 - h2) * x
			res = self.nLLeval_2K(h2_1=h2_1, i_up=i_up, i_G1=i_G1, UW=UW, UUW=UUW, h2=h2, **kwargs)
            
			if (resmin[0] is None) or (res['nLL'] < resmin[0]['nLL']):
				resmin[0] = res
			t1 = time.time()
			#print "one objective function call took %.2f seconds elapsed" % (t1-t0)
			#import pdb; pdb.set_trace()
			return res['nLL']
		if verbose: print "finda2"
		min = minimize1D(f=f, nGrid=nGridA2, minval=minA2, maxval=maxA2,verbose=False)
		#print "numcalls to innerLoopTwoKernel= " + str(self.numcalls)
		return resmin[0]

	def findH2_2K(self, nGridH2=10, minH2=0.0, maxH2=0.99999, nGridA2=10, minA2=0.0, maxA2=1.0, i_up=None, i_G1=None, UW=None, UUW=None, **kwargs):
		'''
		Find the optimal h2 and a2 for a given K (and G1 - if provided in W).
		(default maxH2 value is set to a value smaller than 1 to avoid loss of positive definiteness of the final model covariance)

		Allows to provide a second "low-rank" kernel matrix in form of a rotated design matrix W
		second kernel K2 = W.dot(W.T))

		W may hold a design matrix G1 of a second kernel and some columns that are identical to columns of the design matrix of the first kernel to enable subtracting out sub kernels (as for correcting for proximal contamination)

		Args:
		    nGridH2 : number of h2-grid points to evaluate the negative log-likelihood at. Number of grid points for Brent search intervals (default: 10)
		    minH2   : minimum value for h2 optimization
		    maxH2   : maximum value for h2 optimization
		    nGridA2 : number of a2-grid points to evaluate the negative log-likelihood at. Number of grid points for Brent search intervals (default: 10)
		    minA2   : minimum value for a2 optimization
		    maxA2   : maximum value for a2 optimization
		    i_up    : indices of columns in W corresponding to columns from first kernel that are subtracted of
		    i_G1    : indeces of columns in W corresponding to columns of the design matrix for second kernel G1
		    UW      : U.T.dot(W), where W is [N x S2] np.array holding the design matrix of the second kernel
		    UUW     : W - U.dot(U.T.dot(W))     (provide None if U is full rank)

		Returns:
		    dictionary containing the model parameters at the optimal h2 (and a2 if a G1 is provided in W)
		'''
		#f = lambda x : (self.nLLeval(h2=x,**kwargs)['nLL'])
		if self.Y.shape[1] > 1:
			print "not implemented"
			raise NotImplementedError("only single pheno case implemented")
		resmin = [None]
		noG1 = True
		if i_G1.any():
			noG1 = False
		def f(x,resmin=resmin,**kwargs):
			if noG1:
				res = self.nLLeval_2K(h2_1=0.0, i_up=i_up, i_G1=i_G1, UW=UW, UUW=UUW, h2=x, **kwargs)
			else:
				res = self.findA2_2K(h2=x, i_up=i_up, i_G1=i_G1, UW=UW, UUW=UUW, nGridA2=nGridA2, minA2=minA2, maxA2=maxA2, **kwargs)
			if (resmin[0] is None) or (res['nLL'] < resmin[0]['nLL']):
				resmin[0] = res
			return res['nLL']
		min = minimize1D(f=f, nGrid=nGridH2, minval=minH2, maxval=maxH2)
		return resmin[0]
        
	def find_log_delta(self, sid_count=1, min_log_delta=-5, max_log_delta=10, nGrid=10, **kwargs):
		'''
		perform search for optimal log delta (single kernel case)

		Args:
		     sid_count:      number of log delta grid points to evaluate the negative log-likelihood at. Number of columns in design matrix for kernel for normalization (default: 10)
		     min_log_delta:  minimum value for log delta search (default: -5)
		     max_log_delta:  maximum value for log delta search (default: 10)
		     nGrid:          number of grid points for Brent search intervals (default: 10)
		     
		Returns:
		    dictionary containing the model parameters at the optimal log delta
		'''
		#f = lambda x : (self.nLLeval(h2=x,**kwargs)['nLL'])
		resmin = [None]
		#logging.info("starting log_delta search")
		def f(x,resmin=resmin,**kwargs):
		    h2 = 1.0 / (np.exp(x) * sid_count + 1) #We convert from external log_delta to h2 and then back again so that this
		                                           #code is most similar to findH2

		    res = self.nLLeval(h2=h2,**kwargs)
		    if (resmin[0] is None) or (res['nLL'] < resmin[0]['nLL']):
		        resmin[0] = res
		    #logging.info("search\t{0}\t{1}".format(x,res['nLL']))
		    return res['nLL']
		min = minimize1D(f=f, nGrid=nGrid, minval=min_log_delta, maxval=max_log_delta)
		res = resmin[0]
		internal_delta = 1.0 / res['h2'] - 1.0
		ln_external_delta = np.log(internal_delta / sid_count)
		res['log_delta'] = ln_external_delta
		return res

	def findH2(self, nGridH2=10, minH2=0.0, maxH2=0.99999, estimate_Bayes=False, **kwargs):
		'''
		Find the optimal h2 for a given K. Note that this is the single kernel case. So there is no a2.
		(default maxH2 value is set to a value smaller than 1 to avoid loss of positive definiteness of the final model covariance)
		Args:
		    nGridH2 : number of h2-grid points to evaluate the negative log-likelihood at. Number of columns in design matrix for kernel for normalization (default: 10)
		    minH2   : minimum value for h2 optimization (default: 0.0)
		    maxH2   : maximum value for h2 optimization (default: 0.99999)
		    estimate_Bayes: implement me!   (default: False)

		Returns:
		    dictionary containing the model parameters at the optimal h2
		'''
		#f = lambda x : (self.nLLeval(h2=x,**kwargs)['nLL'])
		resmin = [None]
		#logging.info("starting H2 search")
		assert estimate_Bayes == False, "not implemented"
		if self.Y.shape[1] > 1:
			def f(x):
				res = self.nLLeval(h2=x,**kwargs)
				#check all results for local minimum:
				if (resmin[0] is None):
						resmin[0] = {'nLL':res['nLL'],'h2':np.zeros_like(res['nLL'])+res['h2']}
				else:
					for i in xrange(self.Y.shape[1]):
						if (res['nLL'][i] < resmin[0]['nLL'][i]):
							resmin[0]['nLL'][i] = res['nLL'][i]
							resmin[0]['h2'][i] = res['h2']
				#logging.info("search\t{0}\t{1}".format(x,res['nLL']))
				return res['nLL']
			(evalgrid,resultgrid) = evalgrid1D(f, evalgrid = None, nGrid=nGridH2, minval=minH2, maxval = maxH2, dimF=self.Y.shape[1])
			#import ipdb;ipdb.set_trace()
			return resmin[0]
		elif estimate_Bayes:
			def f(x):
				res = self.nLLeval(h2=x,**kwargs)
				#logging.info("search\t{0}\t{1}".format(x,res['nLL']))
				return res['nLL']
			(evalgrid,resultgrid) = evalgrid1D(f, evalgrid = None, nGrid=nGridH2, minval=minH2, maxval = maxH2, dimF=self.Y.shape[1])
			lik = np.exp(-resultgrid)
			evalgrid = lik * evalgrid[:,np.newaxis]

			posterior_mean = evalgrid.sum(0) / lik.sum(0)
			return posterior_mean
		else: 

			def f(x,resmin=resmin):
				res = self.nLLeval(h2=x,**kwargs)
				if (resmin[0] is None) or (res['nLL'] < resmin[0]['nLL']):
					resmin[0] = res
				#logging.info("search\t{0}\t{1}".format(x,res['nLL']))
				return res['nLL'][0]   
			min = minimize1D(f=f, nGrid=nGridH2, minval=minH2, maxval=maxH2)
			#logging.info("search\t{0}\t{1}".format("?",resmin[0]))
			return resmin[0]


	def posterior_h2(self, nGridH2=1000, minH2=0.0, maxH2=0.99999, **kwargs):
		'''
		Find the optimal h2 for a given K. Note that this is the single kernel case. So there is no a2.
		(default maxH2 value is set to a value smaller than 1 to avoid loss of positive definiteness of the final model covariance)
		Args:
		    nGridH2 : number of h2-grid points to evaluate the negative log-likelihood at. Number of columns in design matrix for kernel for normalization (default: 10)
		    minH2   : minimum value for h2 optimization (default: 0.0)
		    maxH2   : maximum value for h2 optimization (default: 0.99999)
		    estimate_Bayes: implement me!   (default: False)

		Returns:
		    dictionary containing the model parameters at the optimal h2
		'''
		#f = lambda x : (self.nLLeval(h2=x,**kwargs)['nLL'])
		resmin = [None]
		#logging.info("starting H2 search")
		assert self.Y.shape[1] == 1, "only works for single phenotype"
		def f(x):
			res = self.nLLeval(h2=x,**kwargs)
			#check all results for local minimum:
			if (resmin[0] is None):
					resmin[0] = {'nLL':res['nLL'],'h2':np.zeros_like(res['nLL'])+res['h2']}
			else:
				for i in xrange(self.Y.shape[1]):
					if (res['nLL'][i] < resmin[0]['nLL'][i]):
						resmin[0]['nLL'][i] = res['nLL'][i]
						resmin[0]['h2'][i] = res['h2']
			#logging.info("search\t{0}\t{1}".format(x,res['nLL']))
			return res['nLL']
		(evalgrid,resultgrid) = evalgrid1D(f, evalgrid = None, nGrid=nGridH2, minval=minH2, maxval = maxH2, dimF=self.Y.shape[1])
		#import ipdb;ipdb.set_trace()
		return resmin[0], evalgrid, resultgrid


	def nLLeval_2K(self, h2=0.0, h2_1=0.0, dof=None, scale=1.0, penalty=0.0, snps=None, UW=None, UUW=None, i_up=None, i_G1=None, subset=False):
		'''
		TODO: rename to nLLeval

		currently h2 is a scalar, but could be modified to be a vector (i.e., a separate h2 for each phenotype); only if-then-elses need to be modified

		evaluate -ln( N( y | X*beta , sigma^2(h2*K + h2_1*G1*G1.T (1-h2-h2_1)*I ) )),
		where h2>0, h2_1>=0, h2+h2_1 <= 0.99999

		If scale is not equal to 1, then the above is generalized from a Normal to a multivariate Student's t distribution.

		Allows to provide a second "low-rank" kernel matrix in form of a rotated design matrix W
		second kernel K2 = W.dot(W.T))

		G1 is provided as columns in W.
		W is provided in rotated form: UW = U.T*W and UUW =  (W - U*U.T*W)
		W may hold a design matrix G1 of a second kernel and some columns that are identical to columns of the design matrix of the first kernel 
		to enable subtracting out sub kernels (as for correcting for proximal contamination) -- see i_up and i_G1 below.
		UW and UUW can be obtainted by calling rotate on W

		(nice interface wrapper for nLLcore)
		
		Args:
		    h2      : mixture weight between K and Identity (environmental noise)
		    REML    : boolean
					if True   : compute REML
					if False  : compute ML
		    dof     : Degrees of freedom of the Multivariate student-t
						(default None uses multivariate Normal likelihood)
		    scale   : Scale parameter that multiplies the shape matrix in Student's multivariate t (default 1.0, corresponding to a Gaussian)
		    penalty : L2 penalty for the fixed-effect SNPs being tested (default: 0.0)
		    snps    : [N x S] np.array holding S SNPs for N individuals to be tested
		    i_up    : indices of columns in W corresponding to columns from first kernel that are subtracted off
		    i_G1    : indices of columns in W corresponding to columns of the design matrix for second kernel G1
		    UW      : U.T.dot(W), where W is [N x S2] np.array holding the design matrix of the second kernel
		    UUW     : W - U.dot(U.T.dot(W))     (provide None if U is full rank)
		    subset  : if G1 is a subset of G, then we don't need to subtract and add separately (default: False) 
		Returns:
		    Output dictionary:
		    'nLL'       : negative log-likelihood
			'sigma2'    : the model variance sigma^2 (sigma^2_g+sigma^2_g1+sigma^2_e)
			'beta'      : [D*1] array of fixed effects weights beta
			'h2'        : mixture weight between Covariance and noise
			'REML'      : True: REML was computed, False: ML was computed
			'dof'       : Degrees of freedom of the Multivariate student-t
						(default None uses multivariate Normal likelihood)
			'scale'     : Scale parameter that multiplies the Covariance matrix (default 1.0)
		'''

		N = self.Y.shape[0] - self.linreg.D
        
		P = self.Y.shape[1]
		S,U = self.getSU()
		k = S.shape[0]
		if (h2 < 0.0) or (h2 + h2_1 >= 0.99999) or (h2_1 < 0.0):
			return {'nLL':3E20,
					'h2':h2,
					'h2_1':h2_1,
					'scale':scale}
		Sd = (h2 * self.S + (1.0 - h2 - h2_1)) * scale#?(1.0-h2-h2_1)?
		#Sd = (h2*self.S + (1.0-h2))*scale#?(1.0-h2-h2_1)?
		denom = (1.0 - h2 - h2_1) * scale      # determine normalization factor
		if subset: #if G1 is a complete subset of G, then we don't need to subtract and add separately
			h2_1 = h2_1 - h2

		#UY,UUY = self.getUY()
		#YKY = computeAKA(Sd=Sd, denom=denom, UA=UY, UUA=UUY)
		#logdetK = np.log(Sd).sum()
		#
		#if (UUY is not None):#low rank part
		#    logdetK+=(N-k) * np.log(denom)
        
		if UW is not None:
			weightW = np.zeros(UW.shape[1])
			weightW[i_up] = -h2
			weightW[i_G1] = h2_1
		else:
			weightW = None

		Usnps,UUsnps = None,None
		if snps is not None:
            
			if snps.shape[0] != self.Y.shape[0]:
				#pdb.set_trace()
				print "shape mismatch between snps and Y"
			Usnps,UUsnps = self.rotate(A=snps)
                
		result = self.nLLcore(Sd=Sd, dof=dof, scale=scale, penalty=penalty, UW=UW, UUW=UUW, weightW=weightW, denom=denom, Usnps=Usnps, UUsnps=UUsnps)
		result['h2'] = h2
		result['h2_1'] = h2_1
		return result


	def nLLeval(self, h2=0.0, logdelta=None, delta=None, dof=None, scale=1.0, penalty=0.0, snps=None, Usnps=None, UUsnps=None, UW=None, UUW=None, weightW=None, idx_pheno=None):
		'''
		TODO: rename to be a private function
		This function is a hack to fix a parameterization bug regarding h2_1 parameterization in findA2 or findH2 or innerLoop.
		Also, it is a different way of parameterizing nLLeval_2k (e.g., it implements the delta parameterization).
		
		evaluate -ln( N( y | X*beta , sigma^2(h2*K + h2*(W*diag(weightW)*W^T) + (1-h2)*I ) ))  (in h2 parameterization)
		         -ln( N( y | X*beta , sigma^2(delta*K + I )))  (in delta parameterization, 1 Kernel only)

		If scale is not equal to 1, then the above is generalized from a Normal to a multivariate Student's t distribution.

		Allows to provide a second "low-rank" kernel matrix in form of a rotated design matrix W
		second kernel K2 = W.dot(W.T))

		G1 is provided as columns in W.
		W is provided in rotated form: UW = U.T*W and UUW =  (W - U*U.T*W)
		W may hold a design matrix G1 of a second kernel and some columns that are identical to columns of the design matrix of the first kernel to enable subtracting out sub kernels (as for correcting for proximal contamination)

		To be efficient, W should have more rows (people) than columns (SNPs)


		(nice interface wrapper for nLLcore)

		Args:
		    h2      : mixture weight between K and Identity (environmental noise)
		    REML    : boolean
					if True   : compute REML
					if False  : compute ML
		    dof     : Degrees of freedom of the Multivariate Student-t
						(default None uses multivariate Normal likelihood)
		    logdelta: log(delta) allows to optionally parameterize in delta space
		    delta   : delta     allows to optionally parameterize in delta space
		    scale   : Scale parameter that multiplies the shape matrix in the Student's multivariate t (default 1.0, corresponding to a Gaussian)
		    penalty : L2 penalty for SNP effects (default: 0.0)
		    snps    : [N x S] np.array holding S SNPs for N individuals to be tested
		    Usnps   : [k x S] np.array holding S rotated SNPs (U.T.dot(snps)) for N individuals to be tested, where k is rank of the kernel used
		    UUsnps  : [N x S] np.array holding S rotated SNPs (snps - U.dot(U.T.dot(snps))), None in full rank case (k=N)
		    UW      : U.T.dot(W), where W is [N x S2] np.array holding the design matrix of the second kernel
		    UUW     : W - U.dot(U.T.dot(W))     (provide None if U is full rank)
		    weightW : vector of weights for columns in W 
		    idx_pheno: index of the phenotype(s) to be tested

		Returns:
		    Output dictionary:
		        'nLL'       : negative log-likelihood
			    'sigma2'    : the model variance sigma^2 (if h2 parameterization it is sigma^2_g+sigma^2_e; if delta parameterization it is sigma^2_e)
			    'beta'      : [D*1] array of fixed effects weights beta
			    'h2'        : mixture weight between Covariance and noise
			    'REML'      : True: REML was computed, False: ML was computed
			    'dof'       : Degrees of freedom of the Multivariate student-t
						    (default None uses multivariate Normal likelihood)
		        'scale'     : Scale parameter that multiplies the Covariance matrix (default 1.0)
		--------------------------------------------------------------------------
		'''

		N = self.Y.shape[0] - self.linreg.D	#number of degrees of freedom
		P = self.Y.shape[1]
		S,U = self.getSU()
		k = S.shape[0]

		if logdelta is not None:
			delta = np.exp(logdelta)

		if delta is not None:
			Sd = (self.S + delta) * scale
			denom = delta * scale         # determine normalization factor
			h2 = 1.0 / (1.0 + delta)
			assert weightW is None, 'weightW should be none when used with delta or logdelta parameterization, which support only a single Kernel'
		else:
			Sd = (h2 * self.S + (1.0 - h2)) * scale
			denom = (1.0 - h2) * scale      # determine normalization factor
		if (h2 < 0.0) or (h2 >= 1.0):
			return {'nLL':3E20,
					'h2':h2,
					'scale':scale}
		UY,UUY = self.getUY(idx_pheno = idx_pheno)
		P = UY.shape[1]	#number of phenotypes used
		YKY = computeAKA(Sd=Sd, denom=denom, UA=UY, UUA=UUY)
		#logdetK = np.log(Sd).sum()

		#if (UUY is not None):#low rank part
		#	logdetK+=(N - k) * np.log(denom)

		if (snps is not None) and (Usnps is None):
			assert snps.shape[0] == self.Y.shape[0], "shape missmatch between snps and Y"
			Usnps,UUsnps = self.rotate(A=snps)

		if weightW is not None:
			#multiply the weight by h2
			weightW = weightW * h2#Christoph: fixes bug with h2_1 parameterization in findA2 and/or findH2 and/or innerLoop 

		result = self.nLLcore(Sd=Sd, dof=dof, scale=scale, penalty=penalty, UW=UW, UUW=UUW, weightW=weightW, denom=denom, Usnps=Usnps, UUsnps=UUsnps, idx_pheno=idx_pheno)
		result['h2'] = h2
		return result

	def nLLcore(self, Sd=None, dof=None, scale=1.0, penalty=0.0, UW=None, UUW=None, weightW=None, denom=1.0, Usnps=None, UUsnps=None, idx_pheno=None):
		'''
		evaluate -ln( N( U^T y | U^T X*beta , diag(Sd)^-1 + U^T*W*diag(weightW)*W^T*U)) ),

		--------------------------------------------------------------------------
		Args:
		    Sd      : Diagonal scaling for inverse kernel in rotated space (e.g. Sd = 1.0/(delta*S+1.0))
		    dof     : Degrees of freedom of the Multivariate student-t
						(default None - uses multivariate Normal likelihood)
		    denom   : denominator for low rank part (delta*scale for delta scaling of Sd, (1.0-h2)*scale for h2 scaling)
		    scale   : Scale parameter the multiplies the Covariance matrix (default 1.0)
		    penalty : L2 penalty for SNP effects (default: 0.0)
		    Usnps   : [k x S] np.array holding S rotated SNPs (U.T.dot(snps)) for N individuals to be tested, where k is rank of the kernel used
		    UUsnps  : [N x S] np.array holding S rotated SNPs (snps - U.dot(U.T.dot(snps))), None in full rnak case (k=N)
		    UW      : U.T.dot(W), where W is [N x S2] np.array holding the design matrix of the second kernel
		    UUW     : W - U.dot(U.T.dot(W))     (provide None if U is full rank)
		    weightW : vector of weights for columns in W 
		    idx_pheno: index of the phenotype(s) to be tested

		Returns:
		    Output dictionary:
		        'nLL'       : negative log-likelihood
			    'sigma2'    : the model variance sigma^2
			    'beta'      : [D*1] array of fixed effects weights beta
			    'h2'        : mixture weight between Covariance and noise
			    'REML'      : True: REML was computed, False: ML was computed
			    'dof'       : Degrees of freedom of the Multivariate student-t
						    (default None uses multivariate Normal likelihood)
		        'scale'     : Scale parameter that multiplies the Covariance matrix (default 1.0)
		--------------------------------------------------------------------------
		'''

		N = self.Y.shape[0] - self.linreg.D
        
		S,U = self.getSU()#not used, as provided from outside. Remove???
		k = S.shape[0]
		assert Sd.shape[0] == k, "shape missmatch"

		UY,UUY = self.getUY(idx_pheno = idx_pheno)
		P = UY.shape[1]	#number of phenotypes used
		YKY = computeAKA(Sd=Sd, denom=denom, UA=UY, UUA=UUY)
		logdetK = np.log(Sd).sum()

		if (UUY is not None):#low rank part
			logdetK+=(N - k) * np.log(denom)
        
		if Usnps is not None:
            
			snpsKsnps = computeAKA(Sd=Sd, denom=denom, UA=Usnps, UUA=UUsnps)[:,np.newaxis]
			snpsKY = computeAKB(Sd=Sd, denom=denom, UA=Usnps, UB=UY, UUA=UUsnps, UUB=UUY)
        
		if weightW is not None:
			absw = np.absolute(weightW)
			weightW_nonz = absw > 1e-10
		if (UW is not None and weightW_nonz.any()):#low rank updates
			#pdb.set_trace()
			multsign = False
			absw = np.sqrt(absw)
			signw = np.sign(weightW)
			#make sure that the identity works and if needed remove any W with zero
			#weight:
			if (~weightW_nonz).any():
				weightW = weightW[weightW_nonz]
				absw = absw[weightW_nonz]
				signw = signw[weightW_nonz]
				UW = UW[:,weightW_nonz]
				if UUW is not None:
					UUW = UUW[:,weightW_nonz]
			UW = UW * absw[np.newaxis,:]
			if multsign:
				UW_ = UW * signw[np.newaxis,:]
			if UUW is not None:
				UUW = UUW * absw[np.newaxis,:]
			if multsign:
				UUW_ = UUW * signw[np.newaxis,:]
			num_exclude = UW.shape[1]
            

			#WW = np.diag(1.0/weightW) + computeAKB(Sd=Sd, denom=denom, UA=UW, UUA=UUW,
			#UB=UW, UUB=UUW)
			if multsign:
				WW = np.eye(num_exclude) + computeAKB(Sd=Sd, denom=denom, UA=UW, UUA=UUW, UB=UW_, UUB=UUW_)
			else:
				WW = np.diag(signw) + computeAKB(Sd=Sd, denom=denom, UA=UW, UUA=UUW, UB=UW, UUB=UUW)
            
			# compute inverse efficiently
			[S_WW,U_WW] = la.eigh(WW)
			# compute S_WW^{-1} * UWX
                        
			WY = computeAKB(Sd=Sd, denom=denom, UA=UW, UUA=UUW, UB=UY, UUB=UUY)
			UWY = U_WW.T.dot(WY)
			WY = UWY / np.lib.stride_tricks.as_strided(S_WW, (S_WW.size,UWY.shape[1]), (S_WW.itemsize,0))
			# compute S_WW^{-1} * UWy

			# perform updates (instantiations for a and b in Equation (1.5) of
			# Supplement)
			YKY -= (UWY * WY).sum(0)
            
			if Usnps is not None:
				Wsnps = computeAKB(Sd=Sd, denom=denom, UA=UW, UUA=UUW, UB=Usnps, UUB=UUsnps)
				UWsnps = U_WW.T.dot(Wsnps)
				Wsnps = UWsnps / np.lib.stride_tricks.as_strided(S_WW, (S_WW.size,UWsnps.shape[1]), (S_WW.itemsize,0))

				snpsKY -= UWsnps.T.dot(WY)
				# perform updates (instantiations for a and b in Equation (1.5) of
				# Supplement)
				snpsKsnps -= (UWsnps * Wsnps).sum(0)[:,np.newaxis]
            
			# determinant update
			prod_diags = signw * S_WW
			if np.mod((prod_diags < 0).sum(),2):
				raise FloatingPointError("nan log determinant")
			logdetK += np.log(np.absolute(S_WW)).sum()
            
			########

		if Usnps is not None:
			penalty_ = penalty or 0.0
			assert (penalty_ >= 0.0), "penalty has to be positive"
			beta = snpsKY / (snpsKsnps + penalty_)
			if np.isnan(beta.min()):
			    logging.warning("NaN beta value seen, may be due to an SNC (a constant SNP)")
			    beta[snpsKY==0] = 0.0
			variance_explained_beta = (snpsKY * beta)
                        r2 = YKY[np.newaxis,:] - variance_explained_beta 
			if penalty:
				variance_beta = r2 / (N - 1) * (snpsKsnps / ((snpsKsnps + penalty_) * (snpsKsnps + penalty_)))#note that we assume the loss in DOF is 1 here, even though it is less, so the
                                #variance estimate is conservative, due to N-1 for penalty case
                                variance_explained_beta *= (snpsKsnps/(snpsKsnps+penalty_)) * (snpsKsnps/(snpsKsnps + penalty_))
                        else:
                                variance_beta = r2 / (N - 1) / snpsKsnps
                        fraction_variance_explained_beta = variance_explained_beta / YKY[np.newaxis,:] # variance explained by beta over total variance
                        
		else:
			r2 = YKY
			beta = None
			variance_beta = None
                        variance_explained_beta = None
                        fraction_variance_explained_beta = None

		if dof is None:#Use the Multivariate Gaussian
			sigma2 = r2 / N
			nLL = 0.5 * (logdetK + N * (np.log(2.0 * np.pi * sigma2) + 1))
		else:#Use multivariate student-t
			nLL = 0.5 * (logdetK + (dof + N) * np.log(1.0 + r2 / dof))
			nLL +=  0.5 * N * np.log(dof * np.pi) + SS.gammaln(0.5 * dof) - SS.gammaln(0.5 * (dof + N))
		result = {
                        'nLL':nLL,
                        'dof':dof,
                        'beta':beta,
                        'variance_beta':variance_beta,
                        'variance_explained_beta':variance_explained_beta,
                        'fraction_variance_explained_beta':fraction_variance_explained_beta,
                        'scale':scale
                }
		return result


class Linreg(object):
    """ linear regression class"""
    __slots__ = ["X", "Xdagger", "beta", "N", "D"]

    def __init__(self,X=None, Xdagger=None):
        self.N = 0
        self.setX(X=X, Xdagger=Xdagger)
        
    def setX(self, X=None, Xdagger=None):
        self.beta = None
        self.Xdagger = Xdagger
        self.X = X
        if X is not None:
            self.D = X.shape[1]
        else:
            self.D = 1

    def set_beta(self,Y):
        self.N = Y.shape[0]
        if Y.ndim == 1:
            P = 1
        else:
            P = Y.shape[1]    
        if self.X is None:
            self.beta = Y.mean(0)
        else:        
            if self.Xdagger is None:
                self.Xdagger = la.pinv(self.X)       #SVD-based, and seems fast
            self.beta = self.Xdagger.dot(Y)

    def regress(self, Y):
        self.set_beta(Y=Y)
        if self.X is None:
            RxY = Y - self.beta
        else:
            RxY = Y - self.X.dot(self.beta)
        return RxY

    def predict(self,Xstar):
        return Xstar.dot(self.beta)

def computeAKB(Sd, denom, UA, UB, UUA=None, UUB=None):
	"""
	compute asymmetric squared form

	A.T.dot( f(K) ).dot(B)
	"""
	UAS = UA / np.lib.stride_tricks.as_strided(Sd, (Sd.size,UA.shape[1]), (Sd.itemsize,0))
	AKB = UAS.T.dot(UB)
	if UUA is not None:
	    AKB += UUA.T.dot(UUB) / denom
	return AKB

def computeAKA(Sd, denom, UA, UUA=None):
	"""
	compute symmetric squared form
    
	A.T.dot( f(K) ).dot(A)
	"""
	UAS = UA / np.lib.stride_tricks.as_strided(Sd, (Sd.size,UA.shape[1]), (Sd.itemsize,0))
	AKA = (UAS * UA).sum(0)
	if UUA is not None:
		AKA += (UUA * UUA).sum(0) / denom
	return AKA


def create_dir(filename, is_dir=True):
	if is_dir:
		dirname = filename
	else:
		dirname = os.path.dirname(filename)	
	if (not os.path.exists(dirname)) or (not os.path.isdir(dirname)):
		os.makedirs(dirname)

class GWAS(object):
	def __init__(self, K, test_snps, phenotype, covariates=None, 
	             h2=None, interact_with_snp=None, nGridH2=10, standardizer=None, add_bias=True):
		if standardizer is None:
			standardizer = pysnptools.standardizer.Unit()
		self.standardizer = standardizer
		self.phenotype = check_pheno_format(phenotype=phenotype)
		self.covariates = check_pheno_format(phenotype=covariates)
		if add_bias and (self.covariates is None):
			pass # this case is treated in LMM()
		elif add_bias:
			bias = pd.Series(np.ones((self.covariates.shape[0])), index=self.covariates.index)
			self.covariates['bias'] = bias
		elif (add_bias==False) and self.covariates is None:
			raise NotImplementedError("currently a model with neither bias term, nor covariates is not supported.")
		self.snps = test_snps
		self.interact_with_snp = interact_with_snp
		self.nGridH2 = nGridH2
		if self.covariates is not None:
			self.lmm = LMM(X=self.covariates.values, Y=None, G=None, K=K, inplace=True)
		else:
			self.lmm = LMM(X=None, Y=None, G=None, K=K, inplace=True)
		self._find_h2()
	

	def test_snps(self, block_size=None, temp_dir=None):
		result = {}

		if temp_dir is not None:
			if (not os.path.exists(temp_dir)) or (not os.path.isdir(temp_dir)):
				os.makedirs(temp_dir)
		self._generate_intervals(block_size=block_size, total_snps=self.snps.sid.shape[0])
		for stop in range(1,len(self.intervals)):
			res = self.test_snps_block(block_start=self.intervals[stop-1],block_end=self.intervals[stop])
			if temp_dir is not None:
				self.save_results_block(dir_name=temp_dir, res=res, intervals=self.intervals, idx_interval=stop)
			else:
				for p in res.keys():
					if result.has_key(p):
						result[p] = pd.concat([result[p],res[p]])
					else:
						result[p] = res[p]
		return result

	def save_results_block(self, dir_name, res, intervals=None, idx_interval=None):
		for p in res.keys():
			mydir = dir_name + "/" + str(p) +"/"
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
			myfile = "%s%s_block_%0*d.csv" % (mydir, p, fill, stop)
			res[p].to_csv(myfile)

	def test_snps_block(self, block_start, block_end):
		snps = self.snps[:,block_start:block_end].read().standardize(self.standardizer)
		result = {}
		for i, p in enumerate(self.phenotype.columns):
			h2 = self.h2[p]
			self.lmm.setY(self.phenotype[p][:,np.newaxis])		
			res = self.lmm.nLLeval(h2=h2, dof=None, scale=1.0, penalty=0.0, snps=snps.val)
			result[p] = self._format_gwas_results(res=res, snps=snps, h2=h2)
		return result

	@property
	def snp_count(self):
		return self.snps.sid_count

	@property
	def sample_count(self):
	    return self.snps.iid_count
	
	def _find_h2(self):
		self.h2 = {}
		for i, p in enumerate(self.phenotype.columns):
			self.lmm.setY(self.phenotype[p][:,np.newaxis])		
			self.h2[p] = self.lmm.findH2(nGridH2=self.nGridH2)['h2']

	def _generate_intervals(self, block_size, total_snps):
		if block_size is None:
			block_size = total_snps
		intervals = range(0,total_snps,block_size)
		
		if intervals[-1]!= total_snps:
			intervals.append(total_snps)
		self.intervals = intervals

	def _format_gwas_results(self, res, snps, h2):
		beta = res['beta']
		    
		chi2stats = beta*beta/res['variance_beta']
		#p_values = st.chi2.sf(chi2stats,1)[:,0]
		p_values = st.f.sf(chi2stats,1,self.lmm.U.shape[0]-3)[:,0]#note that G.shape is the number of individuals and 3 is the number of fixed effects (covariates+SNP)

		items = [
			('SNP', snps.sid),
			('Chr', snps.pos[:,0]), 
			('GenDist', snps.pos[:,1]),
			('ChrPos', snps.pos[:,2]), 
			('PValue', p_values),
			('SnpWeight', beta[:,0]),
			('SnpWeightSE', np.sqrt(res['variance_beta'][:,0])),
			('SnpFractVarExpl', np.sqrt(res['fraction_variance_explained_beta'][:,0])),
			('Nullh2', np.zeros((snps.sid_count)) + h2)
		]
		return pd.DataFrame.from_items(items)


def check_pheno_format(phenotype):
	if type(phenotype) == np.ndarray:
		if len(phenotype.shape)==1:
			phenotype = phenotype[:,np.newaxis]

		return pd.DataFrame(data=phenotype)
	elif type(phenotype) == pd.DataFrame:
		return phenotype
	elif type(phenotype) == dict:
		assert phenotype.has_key("iid"), "assuming format to be the fastlmm format."
		assert phenotype.has_key("header"), "assuming format to be the fastlmm format."
		assert phenotype.has_key("vals"), "assuming format to be the fastlmm format."
		pheno_new = pd.DataFrame(data=phenotype["vals"], columns=phenotype["header"], index=phenotype["iid"])
		return pheno_new
	elif phenotype is None:
		return None
	else:
		raise NotImplementedError("phenotype is assumed to be a numpy.ndarray or pandas.DataFrame, found %s instead" % (str(type(phenotype))))

def _snp_fixup(snp_input, iid_source_if_none=None):
    if isinstance(snp_input, str):
        return Bed(snp_input)
    elif snp_input is None:
        return iid_source_if_none[:,0:0] #return snpreader with no snps
    else:
        return snp_input

def _pheno_fixup(pheno_input, iid_source_if_none=None):
    if isinstance(pheno_input, str):
        return pysnptools.util.pheno.loadPhen(pheno_input) #!!what about missing=-9?

    if pheno_input is None:
        ret = {
        'header':[],
        'vals': np.empty((iid_source_if_none['vals'].shape[0], 0)),
        'iid':iid_source_if_none['iid']
        }
        return ret

    if len(pheno_input['vals'].shape) == 1:
        ret = {
        'header' : pheno_input['header'],
        'vals' : np.reshape(pheno_input['vals'],(-1,1)),
        'iid' : pheno_input['iid']
        }
        return ret

    return pheno_input

if __name__ == "__main__":
	import logging
	from pysnptools.snpreader import Bed
	import pysnptools.standardizer
	import pysnptools
	import pysnptools.util
	import pysnptools.util.pheno
	import time

	bed_fn = "../../test/data/plinkdata/toydata"
	pheno_fn = bed_fn + ".phe6"#"../../test/data/plinkdata/toydata.phe"
	covariate_fn = 	bed_fn + ".phe"

	
	block_size = 20000
	snp_reader = Bed(bed_fn)#[:,0:50000]
	#snp_reader.run_once()
	if 1:
		standardizer = pysnptools.standardizer.Unit()
	else:
		standardizer = pysnptools.standardizer.Beta(1,2)
	pheno = pysnptools.util.pheno.loadPhen(filename=pheno_fn,   missing ='-9')
	pheno = _pheno_fixup(pheno)
	covariates = _pheno_fixup(covariate_fn, iid_source_if_none=pheno)
	print "intersecting data"
	t00 = time.time()
	snp_intersect,pheno_intersect = pysnptools.util.intersect_apply([snp_reader, pheno], sort_by_dataset=True)
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
		K = snp_intersect.kernel(standardizer=standardizer,blocksize=block_size)
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
		print "running GWAS"
		t0 = time.time()
		mygwas = GWAS(K=K, test_snps=snp_intersect, phenotype=pheno, covariates=covariates, h2=None, interact_with_snp=None, nGridH2=10, standardizer=standardizer)
		if 1:
			result = mygwas.test_snps(block_size=block_size, temp_dir='./temp_dir_testdata/')
		else:
			result_block = mygwas.test_snps_block(block_start=0, block_end=block_size)
			mygwas.save_results_block("./res_check/", result_block)
		t1 = time.time()
		print "done running GWAS after %.4fs" % (t1-t0)
		print "total: %.4fs" % (t1-t00)

