import scipy as SP
import pdb

def sq_dist(X1,X2=None):
    """
    computes a matrix of all pariwise squared distances
    """
    if X2==None:
        X2 = X1
    assert X1.shape[1]==X2.shape[1], 'dimensions do not match'

    n = X1.shape[0]
    m = X2.shape[0]
    d = X1.shape[1]
    # (X1 - X2)**2 = X1**2 + X2**2 - 2X1X2
    X1sq = SP.reshape((X1**2).sum(1),n,1)
    X2sq = SP.reshape((X2**2).sum(1),m,1)

    K = SP.tile((X1*X1).sum(1),(m,1)).T + SP.tile((X2*X2).sum(1),(n,1)) - 2*SP.dot(X1,X2.T)
    return K
