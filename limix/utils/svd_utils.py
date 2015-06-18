import numpy.linalg as nla

def svd_reduce(X, tol = 1e-9):
    U, Sh, V = nla.svd(X, full_matrices=0)
    I = Sh < tol
    if I.any():
        #warnings.warn('G has dependent columns, dimensionality reduced')
        Sh = Sgh[~I]
        U  = Ug[:, ~I]
        V  = SP.eye(Sgh.shape[0])
        X  = U * Sh[SP.newaxis,:]
    S = Sh**2
    return X, U, S, V
