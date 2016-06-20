import numpy as np
import numpy.linalg as la

def generate_matrix_normal(R,C):
	N=R.shape[0]
	P=C.shape[0]
	sqrtR = la.cholesky(R)
	sqrtC = la.cholesky(C)
	Y = np.random.randn(N,P)
	Y = np.dot(sqrtR, Y)
	Y = np.dot(Y,sqrtC)
	return Y


def pexph(M=None,eigh=None,exp=0.5,eps=1e-8,symmetric=True,debug=False):
	if eigh is None:
		s,U = la.eigh(M)
	else:
		s,U = eigh
	i_nonz = s>=eps
	if i_nonz.any():
		s=s[i_nonz]
		U=U[:,i_nonz]
		res = U * (s**exp)[np.newaxis,:]
		if symmetric:
			res = res.dot(U.T)
	else:
		if symmetric:
			res = np.zeros_like(M)
		else:
			res = np.zeros((M.shape[0],0))
	if debug:
		M_ = pexph(M,exp=1.0)
		diff = np.absolute(M-M_).sum()
		print(diff)
		assert diff<1E-8
	ret = {
		"res": res,
		"eigh": (s,U)
	}
	return ret

def pexp(M=None,svd=None,exp=0.5, eps=1e-8,symmetric=True,debug=False):
	if svd is None:
		u,s,v = la.svd(M,full_matrices=False)
	else:
		u,s,v = svd
	i_nonz = s>=eps
	if i_nonz.any():
		s=s[i_nonz]
		u=u[:,i_nonz]
		v=v[i_nonz]
		res = u * (s**exp)[np.newaxis,:]
		if symmetric:
			res = res.dot(v)
	else:
		if symmetric:
			res = np.zeros_like(M)
		else:
			res = np.zeros((M.shape[0],0)) 
	if debug:
		M_ = pexp(M,exp=1.0)
		diff = np.absolute(M-M_).sum()
		print(diff)
		assert diff<1E-8
	ret = {
		"res": res,
		"svd": (u,s,v)
	}
	return ret

def compute_XYA(DY, X, A=None):

	if A is not None:#general case
		DYA = DY.dot(A.T)
	else:#any effect
		DYA = DY
	return X.T.dot(DYA)#should be pre-computed


def vec(X):
	res = np.empty((X.shape[0]*X.shape[1]))
	for i in range(X.shape[1]):
		res[i*X.shape[0]:(i+1)*X.shape[0]] = X[:,i]
	return res

def compute_X1KX2(Y, D, X1, X2, A1=None, A2=None):
	#import ipdb; ipdb.set_trace()
	R,C = Y.shape
	if A1 is None:
		nW_A1 = Y.shape[1]	  			#A1 = np.eye(Y.shape[1])	#for now this creates A1 and A2
	else:
		nW_A1 = A1.shape[0]

	if A2 is None:
		nW_A2 = Y.shape[1]
		#A2 = np.eye(Y.shape[1])	#for now this creates A1 and A2
	else:
		nW_A2 = A2.shape[0]
			
	nW_X1 = X1.shape[1]
	rows_block = nW_A1 * nW_X1

	if 0:#independentX2:
		nW_X2 = 1
	else:
		nW_X2 = X2.shape[1]
	cols_block = nW_A2 * nW_X2
	block = np.zeros((rows_block,cols_block))


	if (R>C) or (A1 is None) or (A2 is None):
		for c in range(C):
			X1D = X1 * D[:,c:c+1]
			X1X2 = X1D.T.dot(X2)
			if (A1 is None) and (A2 is None):
				block[c*X1.shape[1]:(c+1)*X1.shape[1], c*X2.shape[1]:(c+1)*X2.shape[1]] += X1X2
			elif (A1 is None):
				block[c*X1.shape[1]:(c+1)*X1.shape[1],:] += np.kron(A2[:,c:c+1].T,X1X2)
			elif (A2 is None):
				block[:,c*X2.shape[1]:(c+1)*X2.shape[1]] += np.kron(A1[:,c:c+1],X1X2)
			else:
				A1A2 = np.outer(A1[:,c],A2[:,c])
				block += np.kron(A1A2,X1X2)
	else:
		for r in range(R):
			A1D = A1 * D[r:r+1,:]
			A1A2 = A1D.dot(A2.T)
			X1X2 = X1[r,:][:,np.newaxis].dot(X2[r,:][np.newaxis,:])
			block += np.kron(A1A2,X1X2)

	return block