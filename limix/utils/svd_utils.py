import numpy.linalg as nla
import scipy as sp

def svd_reduce(X, tol = 1e-9):
    U, Sh, V = nla.svd(X, full_matrices=0)
    I = Sh < tol
    if I.any():
        #warnings.warn('G has dependent columns, dimensionality reduced')
        Sh = Sh[~I]
        U  = U[:, ~I]
        V  = sp.eye(Sh.shape[0])
        X  = U * Sh[sp.newaxis,:]
    S = Sh**2
    return X, U, S, V
