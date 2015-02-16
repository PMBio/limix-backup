import scipy.linalg as la
import numpy as np

class psd_solver(object):
    """description of class"""

    def __init__(self, A, lower=True, threshold=1e-10,check_finite=True,overwrite_a=False):
        self._s=None
        self._U=None
        self._chol=None
        self._lower=lower
        try:
            self._chol=la.cholesky(A,overwrite_a=overwrite_a,check_finite=True).T
    
        except la.LinAlgError:
            s,U = la.eigh(A,lower=lower)
            i_pos = (s>threshold)
            self._s = s[i_pos]
            self._U = U[:,i_pos]
            

            
    def solve(self,b,overwrite_b=False,check_finite=True):
        """
        solve A \ b
        """
        if self._s is not None:
            res = self._U.T.dot(b)
            res /= self._s[:,np.newaxis]
            res = self._U.dot(res)
        elif self._chol is not None:
            res = la.cho_solve((self._chol,self._lower),b=b,overwrite_b=overwrite_b,check_finite=check_finite)

        return res

