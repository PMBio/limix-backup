from limix.core.type.cached import Cached, cached
from limix.core.type.observed import Observed

import sys
import numpy.linalg as la 
import numpy as np
import scipy as sp
import limix.core.association.kron_util as kron_util
import mingrid
#import fastlmm.util.mingrid as mingrid
import time

class KroneckerLMM(Cached):
	def __init__(self, Y, R1, C1, R2, C2, X, A=None, h2=0.5, reml=True):
		"""
		Kronecker mixed model implementation

		model specification:

		vec(Y) ~ N( vec(sum_i vec(X_i * B_i * A_i ), sigma2 * [ h2 * C1 x R1 + (1-h2) * C2 x R2 ]  )

		Args:
			Y:		phenotypes [N x P] ndarray
			R1:		first row covariance matrix [N x N] ndarray
			C1:		first column covariance matrix [P x P] ndarray
			R2:		second row covariance matrix [N x N] ndarray
			C2:		second column covariance matrix [P x P] ndarray
			X:		list of row covariates matrices of length T, each [N x D_t] ndarray
			A:		list of column covariates matrices of length T, each either [dof_t x P] ndarray or None.
					None will result in an efficient implementation of any effect.
			h2:		heritability (default 0.5)
			reml:	use REML (True) or ML (False)? (default: True)
		"""
		Cached.__init__(self)
		C = []
		R = []
		C.append(C1)
		C.append(C2)
		R.append(R1)
		R.append(R2)
		self.C = C
		self.R = R
		if type(X) is list:
			self.X = X
		else:
			self.X = [X]
		if type(A) is list:
			self.A = A
		else:
			self.A = [A]	
		assert len(self.X)==len(self.A), "missmatch between len(X)=%i and len(A)=%i" % (len(self.X), len(self.A))
		self.h2 = h2
		self.Y=Y
		self.reml=reml

	@property
	def reml(self):
		"""
		Boolean indicator if REML is used instead of ML
		"""
		return self._reml

	@reml.setter
	def reml(self,value):
		"""
		Boolean indicator if REML is used instead of ML
		"""
		self.clear_cache("reml")
		self._reml=value
	
	@property
	def h2(self):
		"""
		heritability parameter in the covariance term
		sigma2 * ( h2 * C1 x R1 + (1-h2) * C2 x R2 )
		with 0<=h2<1
		"""
		return self._h2

	@h2.setter
	def h2(self, value):
		"""
		heritability parameter in the covariance term
		sigma2 * ( h2 * C1 x R1 + (1-h2) * C2 x R2 )
		with 0<=h2<1
		"""
		assert value>=0.0, "h2 has to be non-negative, found %f" % value
		assert value<1.0, "h2 has to be smaller than 1.0, found %f" % valyue
		self.clear_cache("h2")
		self._h2 = value

	@property
	def Y(self):
		"""
		phenotypes [N x P] ndarray
		"""
		return self._Y

	@Y.setter
	def Y(self, value):
		"""
		phenotypes [N x P] ndarray
		"""
		self.clear_cache("Y")
		self._Y = value

	@property
	def C(self):
		"""
		list of column covariances (length = 2)
		[C1, C2]
		"""
		return self._C

	@C.setter
	def C(self, value):
		"""
		list of column covariances (length = 2)
		[C1, C2]
		"""
		assert len(value)==2, "length missmatch C, need 2 found %i" % len(value)
		self.clear_cache("C")
		self._C = value

	@property
	def R(self):
		"""
		list of row covariances (length = 2)
		[R1, R2]
		"""
		return self._R

	@R.setter
	def R(self, value):
		"""
		list of row covariances (length = 2)
		[R1, R2]
		"""
		assert len(value)==2, "length missmatch R, need 2 found %i" % len(value)
		self.clear_cache("R")
		self._R = value

	@property
	def A(self):
		"""
		list of column fixed effect matrices
		"""
		return self._A

	@A.setter
	def A(self, value):
		"""
		list of column fixed effect matrices
		"""
		self.clear_cache("A")
		if type(value) is list:
			self._A = value
		else:
			self._A = [value]

	@property
	def X(self):
		"""
		list of row fixed effect matrices
		"""
		return self._X

	@X.setter
	def X(self, value):
		"""
		list of row fixed effect matrices
		"""
		self.clear_cache("X")
		if type(value) is list:
			self._X = value
		else:
			self._X = [value]

	@property
	def P(self):
		"""
		number of traits (=columns of Y)
		"""
		return self.C[0].shape[0]

	@property
	def N(self):
		"""
		number of samples (=rows of Y)
		"""
		return self.R[0].shape[0]

	@property
	def length(self):
		"""
		number of fixed effect terms
		"""
		assert len(self.X)==len(self.A), "missmatch between len(X)=%i and len(A)=%i" % (len(X), len(A))
		return len(self.X)
	
	def dof_X(self, X):
		"""
		The number of columns of the row fixed effects design matrices X.

		Returns:
			1-dimensional ndarray of columns of X for each fixed effect term
		"""	
		if type(X) is list:
			dof = np.empty((len(X)),dtype=np.int)
			for i in xrange(len(X)):
				dof[i] = X[i].shape[1]
			return dof
		else:
			return X.shape[1]

	def dof_A(self, A):
		"""
		The number of rows of the column fixed effects design matrices A.

		Returns:
			1-dimensional ndarray of rows of A for each fixed effect term
		"""	
		if type(A) is list:
			dof = np.empty((len(A)),dtype=np.int)
			for i in xrange(len(A)):
				if A[i] is None:
					dof[i] = self.P
				else:
					dof[i] = A[i].shape[0]
			return dof
		elif A is None:
			return self.P
		else:
			return A.shape[0]

	@property
	def dof(self):
		"""
		The degrees of freedom of the fixed effects. (=number of entries in beta)

		Returns:
			1-dimensional ndarray of degrees of freedom for each fixed effect term
		"""
		dof = self.dof_A(A=self.A) * self.dof_X(X=self.X)
		return dof

	@cached(["C","R","h2"])
	def D_rot(self):
		"""
		computes the diagonal result of the diagonalized covariance matrix
		"""
		sR = self.Rrot()[0][0]
		sC = self.Crot()[0][0]
		return 1.0 / (self.h2 * sR[:,np.newaxis] * sC[np.newaxis,:] + (1.0-self.h2))

	def covariance_vec(self, i):
		"""
		computes the possibly large covariance matrix of vec(Y)
		"""
		return np.kron(self.C[i],self.R[i])

	@cached(["R"])
	def Rrot(self):
		"""
		row rotation matrix
		"""
		return KroneckerLMM.rot_kron(self.R[0], self.R[1])#, h2=self.h2)

	@cached(["C"])
	def Crot(self):
		"""
		column rotation matrix
		"""
		return KroneckerLMM.rot_kron(self.C[0], self.C[1])#, h2=self.h2)

	@cached(["Y","C","R"])
	def Yrot(self):
		"""
		computes the rotated target matrix Y (Rrot * Y * Crot)
		
		Returns:
			rotated target matrix Y (ndarray)
		"""
		res = self.Rrot()[0][1].T.dot(self.Y).dot(self.Crot()[0][1])
		return res

	def rotate_X(self, X):
		"""
		internal function to rotate a single row design matrix 
		"""
		return self.Rrot()[0][1].T.dot(X)

	def rotate_A(self, A):
		"""
		internal function to rotate a single column design matrix
		"""
		if A is None:
			#return self.Crot()[1]
			#return np.eye(self.P)
			return None
		else:
			return A.dot(self.Crot()[0][1])

	@cached(["X","R"])
	def X_rot(self):
		"""
		compute the rotated versions of the row fixed effect design matrices.
		The matrices are rotated with the column rotation matrix as computed by self.Rrot() (= KroneckerLMM.rot_kron(R1, R2)).

		Returns:
			list of rotated row fixed effect design matrices
		"""
		res = []
		for i in xrange(self.length):
			res.append(self.rotate_X(X=self.X[i]))
		return res

	@cached(["A","C"])
	def A_rot(self):
		"""
		compute the rotated versions of the column fixed effect design matrices.
		The matrices are rotated with the column rotation matrix as computed by self.Crot() (=KroneckerLMM.rot_kron(C1, C2)).

		Returns:
			list of rotated column fixed effect design matrices
		"""
		res = []
		for i in xrange(self.length):
			res.append(self.rotate_A(A=self.A[i]))
		return res

	@cached(["X","A","C","R","h2"])
	def XKX(self):
		"""
		The Hessian matrix of the fixed effect weights beta
		
		Returns:
			the cross product of the fixed effects with themselves
		"""
		dof = self.dof
		dof_cumsum = np.concatenate(([0],dof.cumsum()))
		XKX = np.empty((dof.sum(),dof.sum()))
		for i in xrange(self.length):
			XKX[dof_cumsum[i]:dof_cumsum[i+1],dof_cumsum[i]:dof_cumsum[i+1]] = kron_util.compute_X1KX2(Y=self.Y, D=self.D_rot(), A1=self.A_rot()[i], A2=self.A_rot()[i], X1=self.X_rot()[i], X2=self.X_rot()[i])
			for j in xrange(i+1,self.length):
				XKX[dof_cumsum[i]:dof_cumsum[i+1],dof_cumsum[j]:dof_cumsum[j+1]] = kron_util.compute_X1KX2(Y=self.Y, D=self.D_rot(), A1=self.A_rot()[i], A2=self.A_rot()[j], X1=self.X_rot()[i], X2=self.X_rot()[j])
				XKX[dof_cumsum[j]:dof_cumsum[j+1],dof_cumsum[i]:dof_cumsum[i+1]] = XKX[dof_cumsum[i]:dof_cumsum[i+1],dof_cumsum[j]:dof_cumsum[j+1]].T
		return XKX

	@cached(["X","A","C","R","Y","h2"])
	def XKY(self):
		"""
		Returns:
			the cross product between the fixed effects and the target Y
		"""
		dof = self.dof
		dof_cumsum = np.concatenate(([0],dof.cumsum()))
		XKY = np.empty((dof.sum(),1))
		DY = self.D_rot() * self.Yrot()
		for i in xrange(self.length):
			XKY[dof_cumsum[i]:dof_cumsum[i+1],0] = kron_util.vec(kron_util.compute_XYA(DY=DY, X=self.X_rot()[i], A=self.A_rot()[i]))		
		return XKY

	@cached(["C","R","Y","h2"])
	def YKY(self):
		"""
		Returns:
			the total variance of Y
		"""
		YKY = (self.Yrot() * self.Yrot() * self.D_rot()).sum()
		return YKY

	@cached(["X","A","Y","C","R","h2"])
	def beta(self):
		"""
		Returns:
			the maximum likelihood weight vector
		"""
		XKX = self.XKX()
		XKy = self.XKY()
		beta = la.solve(XKX,XKy)
		return beta
	
	@cached(["X","A","Y","C","R","h2"])
	def resKres(self):
		"""
		Returns:
			the residual sum of squares
		"""
		yKy = self.YKY()
		beta = self.beta()
		XKy = self.XKY()
		rss = yKy - (beta*XKy).sum()
		return rss

	@cached(["C","R","h2"])
	def logdet_K(self):
		"""
		computes the log determinant of the unscaled model covariance (h2 * C1 x R1 + (1-h2) * C2 x R2)
		Note that we do not take the factor sigma2 into account.

		Returns:
			log determinant of the covariance
		"""
		D = self.D_rot()
		logD = np.log(D)
		logdet_D = -logD.sum()
		logdet_R = self.P * np.log(self.Rrot()[1][0]).sum()
		logdet_C = self.N * np.log(self.Crot()[1][0]).sum()
		logdet = logdet_D + logdet_R + logdet_C
		return logdet

	@cached(["X","A","C","R","h2"])
	def logdet_XKX(self):
		"""
		computes the log determinant of the Hessian (XKX) of the fixed effect weights

		Returns:
			log determinant of XKX
		"""
		XKX = self.XKX()
		sign,logdet_XKX = la.slogdet(XKX)
		return logdet_XKX

	@cached(["X","A","Y","C","R","h2","reml"])
	def LL(self):
		"""
		compute the log likelihood of the model

		Returns:
			log likelihood
		"""
		yKy = self.resKres()
		logdet = self.logdet_K()
		if self.reml:
			logdet += self.logdet_XKX()
			dof = self.dof.sum()
			var = yKy/(self.N*self.P-dof)
			const = (self.N*self.P-dof) * (np.log(2.0*sp.pi) + np.log(var))
		else:
			var = yKy/(self.N*self.P)
			const = self.N*self.P * (np.log(2.0*sp.pi) + np.log(var))
		dataterm = yKy/var
		logl =  -0.5 * (const + logdet + dataterm)
		return logl

	def find_h2(self, nGridH2=10, minH2=0.0, maxH2=0.99999, verbose=False):
		'''
		Finds the optimal kernel mixture weight h2 (heritability) and returns the log-likelihood
		
		(default maxA2 value is set to less than 1 as loss of positive definiteness of the final model covariance depends on h2)
		
		Args:
			nGridH2 : number of h2-grid points to evaluate the negative log-likelihood at. Number of grid points for Brent search intervals (default: 10)
			minH2   : minimum value for h2 optimization
			maxH2   : maximum value for h2 optimization
			verbose : verbose output? (default: False)
			
		Returns:
			max log likelihood obtained
		'''
		
		self.numcalls = 0
		resmin = [None]
		def f(x,resmin=resmin, **kwargs):
			self.numcalls+=1
			t0 = time.time()
			self.h2 = x
			res = -self.LL()
			
			if (resmin[0] is None) or (res < resmin[0]['nLL']):
				resmin[0] = {
				'nLL':res,
				'h2':x
				}
			t1 = time.time()
			if verbose:
				print "one objective function call took %.2f seconds" % (t1-t0)
			return res
		if verbose:
			print "findh2"
		minimum = mingrid.minimize1D(f=f, nGrid=nGridH2, minval=minH2, maxval=maxH2, verbose=verbose)
		if verbose:
			print "numcalls to log likelihood= " + str(self.numcalls)
		self.h2 = resmin[0]['h2']
		return -resmin[0]['nLL']
	
	@staticmethod
	def rot_kron(C1, C2):
		"""
		static method to compute the joint diagonalization of covariance matrices C1 and C2.

		Returns:
			rotation matrix
		"""
		res_C2 = kron_util.pexph(C2,exp=-0.5)
		sqrtC2i = res_C2["res"]
		C1_rot = sqrtC2i.T.dot(C1).dot(sqrtC2i.T)
		s,U = la.eigh(C1_rot)
		res = (s,sqrtC2i.dot(U))
		ret = [res,res_C2["eigh"]]
		return ret

