import numpy as np
import scipy.linalg as la

class LmmKronecker(object):
	pass

	def __init__(self, X=None, A=None, Y=None, K=None, inplace=False):
		'''
		Input:
		forcefullrank   : if True, then the code always computes K and runs cubically
							(False)
		'''
		self.numcalls = 0
		self.setX(X=X, regressX=regressX, linreg=linreg)    #set the covariates (needs to be first)
		self.forcefullrank = forcefullrank
		self.setK(K=K, G=G, inplace=inplace)                 #set the kernel, if available
		self.setY(Y=Y)     

	def setY(self, Y):
		'''
		set the phenotype Y.
		--------------------------------------------------------------------------
		Input:
		Y       : [NxP] dimensional array of phenotype values
		--------------------------------------------------------------------------
		'''
		self.Y = Y
		self.UY = None


	def setX(self, X=None, A=None):
		self.X = X
		self.A = A
		self.UX = None
		self.XKX = None



	def setK(self, R1=None, C1=None, R2=None, C2=None, inplace=False):
		'''
		set the Covariance matrices.
		--------------------------------------------------------------------------
		Input:
		R1 : [N*N] array, random effects covariance (positive semi-definite)
		C1 : [N*N] array, random effects covariance (positive semi-definite)
		R2 : [N*N] array, random effects covariance (positive semi-definite)
		C2 : [N*N] array, random effects covariance (positive semi-definite)
		--------------------------------------------------------------------------
		'''
		self.clear_cache()
		if inplace:
			self.R1 = R1
			self.C1 = C1
			self.R2 = R2
			self.C2 = C2
		else:
			self.R1 = R1.copy()
			self.C1 = C1.copy()
			self.R2 = R2.copy()
			self.C2 = C2.copy()

	def clear_cache(self):
		self.U_R = None
		self.U_C = None
		self.S_R = None
		self.S_C = None
		self.UY = None
		self.UX = None #for now assume that X has independent effects
		self.XKX = None
		self.R1 = None
		self.C1 = None
		self.R2 = None
		self.C2 = None

	def nLL(self, snps, Asnps=None):
		nR,nC = Y.shape

		pass


def compute_XYA(Y, D, X, A=None):
	DY = D * Y
	if A is not None:
		DYA = DY.dot(A.T)
	else:
		DYA = DY
	XYA = X1.T.dot(DYA)#should be pre-computed
	return XYA



def compute_D(S_C, S_R, delta=1.0):
	return 1.0 / (delta + np.outer(S_C, S_R))

def ldet_Kron(S_C, S_R, D=None, delta=1.0):
	"""
	compute the log determinant
	"""
	if D is None:
		D = compute_D(S_C=S_C, S_R=S_R, delta=delta)
	ldet = R * np.log(S_R).sum() + C * np.log(S_C).sum() - np.log(D).sum()
	return ldet


def compute_Kronecker_beta(Y, D, X, A):
    n_terms = len(X)
    n_weights1 = 0

    for term1 in xrange(n_terms):
	    if A[term1] is None:		
		    A[term1] = np.eye(Y.shape[1])	#for now this creates A1 and A2

	    assert((X[term1].shape[0])==R);
	    assert((A[term1].shape[1])==C);
	    n_weights1+=A[term1].shape[0] * X[term1].shape[1]
	
    XYA = np.zeros(n_weights)
    n_weights1 = 0
    for term1 in xrange(n_terms):
        n_weights2 = 0
        XYA_block = compute_XYA(Y, D, X, A=None)

        XYA[n_weights1:n_weights1 + A[term1].shape[0] * X[term1].shape[1]] = XYA_block.flatten()
    cov_beta = mean.compute_XKX()

    inv_cov_beta =la.inv(cov_beta)
    betas = np.dot(inv_cov_beta,XYA.flatten())
    #chol_beta = la.cho_factor(a=cov_beta, lower=False, overwrite_A=True, check_finite=True)
    #betas = la.cho_solve(chol_beta,b=XYA.flatten(), overwrite_b=False, check_finite=True)

    total_sos = (Y * DY).sum()
    var_expl = (betas * XYA.flatten()).sum()
	
    sigma2 = (res - var_expl)/(R*C) #change according to REML
    ldet = ldet_Kron(S_C, S_R, D=D, delta=1.0)

    nLL = 0.5 * ( R * C * ( np.log(2.0*np.pi) + log(sigma2) + 1.0) + ldet)

    W=[]
    cum_sum = 0
    for term in xrange(n_terms):
	    current_size = X[term].shape[1] * A[term].shape[0]
	    W_term = np.reshape(betas[cum_sum:cum_sum+current_size], (X[term].shape[1],A[term].shape[0]), order='C')
	    cum_sum += current_size
	    pass
    pass
