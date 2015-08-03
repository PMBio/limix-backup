import scipy as sp

def vei_CoR_veX(X, C=None, R=None):
    """
    Args:
        X:  NxPxS tensor
        C:  CxC row covariance (if None: C set to I_PP)
        R:  NxN row covariance (if None: R set to I_NN)
    Returns:
        NxPxS tensor obtained as ve^{-1}((C \kron R) ve(X))
        where ve(X) reshapes X as a NPxS matrix.
    """
    _X = X.transpose((0,2,1))
    if R is not None:   RV = sp.tensordot(R, _X, (1,0))
    else:               RV = _X
    if C is not None:   RV = sp.dot(RV, C.T)
    return RV.transpose((0,2,1))

