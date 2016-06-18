"""Matrix linear algebra routines needed for GP models"""



import scipy as SP
import numpy as np
import scipy.linalg as linalg
import logging


_MIN_EIGVAL = np.sqrt(np.finfo(float).eps)


def QS_from_K(K):
    global _MIN_EIGVAL

    (S, Q) = np.linalg.eigh(K)
    ok = S >= _MIN_EIGVAL
    S = S[ok]
    Q = Q[:, ok]
    return (Q, S)

def QS_from_G(G):
    (Q, Ssq, _) = np.linalg.svd(G, full_matrices=False)
    S = Ssq**2
    return (Q, S)


def solve_chol(A,B):
    """
    Solve cholesky decomposition::

        return A\(A'\B)

    """
    # X = linalg.solve(A,linalg.solve(A.transpose(),B))
    # much faster version
    X = linalg.cho_solve((A, True), B)
    return X






def jitChol(A, maxTries=10, warning=True):

    """Do a Cholesky decomposition with jitter.

    Description:


    U, jitter = jitChol(A, maxTries, warning) attempts a Cholesky
     decomposition on the given matrix, if matrix isn't positive
     definite the function adds 'jitter' and tries again. Thereafter
     the amount of jitter is multiplied by 10 each time it is added
     again. This is continued for a maximum of 10 times.  The amount of
     jitter added is returned.
     Returns:
      U - the Cholesky decomposition for the matrix.
      jitter - the amount of jitter that was added to the matrix.
     Arguments:
      A - the matrix for which the Cholesky decomposition is required.
      maxTries - the maximum number of times that jitter is added before
       giving up (default 10).
      warning - whether to give a warning for adding jitter (default is True)

    See also
    CHOL, PDINV, LOGDET


    Copyright (c) 2005, 2006 Neil D. Lawrence

    """
    jitter = 0
    i = 0

    while(True):
        try:
            # Try --- need to check A is positive definite
            if jitter == 0:
                jitter = abs(SP.trace(A))/A.shape[0]*1e-6
                LC = linalg.cholesky(A, lower=True)
                return LC.T, 0.0
            else:
                if warning:
                    # pdb.set_trace()
		    # plt.figure()
		    # plt.imshow(A, interpolation="nearest")
		    # plt.colorbar()
		    # plt.show()
                    logging.error("Adding jitter of %f in jitChol()." % jitter)
                LC = linalg.cholesky(A+jitter*SP.eye(A.shape[0]), lower=True)

                return LC.T, jitter
        except linalg.LinAlgError:
            # Seems to have been non-positive definite.
            if i<maxTries:
                jitter = jitter*10
            else:
                raise linalg.LinAlgError("Matrix non positive definite, jitter of " +  str(jitter) + " added but failed after " + str(i) + " trials.")
        i += 1
    return LC


def jitEigh(A,maxTries=10,warning=True):
    """
    Do a Eigenvalue Decomposition with Jitter,

    works as jitChol
    """
    warning = True
    jitter = 0
    i = 0

    while(True):
        if jitter == 0:
            jitter = abs(SP.trace(A))/A.shape[0]*1e-6
            S,U = linalg.eigh(A)

        else:
            if warning:
                # pdb.set_trace()
		# plt.figure()
		# plt.imshow(A, interpolation="nearest")
		# plt.colorbar()
		# plt.show()
                logging.error("Adding jitter of %f in jitEigh()." % jitter)
            S,U = linalg.eigh(A+jitter*SP.eye(A.shape[0]))

        if S.min()>1E-10:
            return S,U

        if i<maxTries:
            jitter = jitter*10
        i += 1
            
    raise linalg.LinAlgError("Matrix non positive definite, jitter of " +  str(jitter) + " added but failed after " + str(i) + " trials.")

