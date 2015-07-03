import scipy as sp
import scipy.linalg as la
import pdb
from limix.core.covar import FreeFormCov
from limix.core.mean import MeanKronSum
from limix.core.gp import GP2KronSum
from limix.core.gp import GP
from limix.utils.preprocess import covar_rescale
import time
import copy

if __name__=='__main__':

    # define phenotype
    N = 1000
    P = 4
    Y = sp.randn(N,P)

    # define fixed effects
    F = []; A = []
    F.append(1.*(sp.rand(N,2)<0.5))
    A.append(sp.eye(P))

    # define row caoriance
    f = 10
    X = 1.*(sp.rand(N, f)<0.2)
    R = covar_rescale(sp.dot(X,X.T))
    R+= 1e-4 * sp.eye(N)
    S_R, U_R = la.eigh(R)

    # define col covariances
    Cg = FreeFormCov(P)
    Cn = FreeFormCov(P)
    Cg.setRandomParams()
    Cn.setRandomParams()

    # define gp and optimize
    gp = GP2KronSum(Y=Y, F=F, A=A, Cg=Cg, Cn=Cn, S_R=S_R, U_R=U_R)
    gp.optimize()

