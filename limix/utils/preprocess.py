import scipy as SP
import scipy.linalg as LA
import scipy.stats as ST

def toRanks(A):
    """
    converts the columns of A to ranks
    """
    AA=SP.zeros_like(A)
    for i in range(A.shape[1]):
        AA[:,i] = ST.rankdata(A[:,i])
    AA=SP.array(SP.around(AA),dtype="int")-1
    return AA

def gaussianize(Y):
    """
    converts the columns of Y to the quantiles of a standard normal
    """
    N,P = Y.shape

    YY=toRanks(Y)
    quantiles=(SP.arange(N)+0.5)/N
    gauss = ST.norm.isf(quantiles)
    Y_gauss=SP.zeros((N,P))
    for i in range(P):
        Y_gauss[:,i] = gauss[YY[:,i]]
    Y_gauss *= -1
    return Y_gauss

def regressOut(Y,X):
    """
    regresses out X from Y
    """
    Xd = LA.pinv(X)
    Y_out = Y-X.dot(Xd.dot(Y))
    return Y_out


