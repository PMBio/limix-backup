import sys
sys.path.insert(0, '../..')
import scipy as sp
import scipy.linalg as la
import pdb
from limix.core.covar import FreeFormCov
from limix.core.covar import KronCov
from limix.core.covar import SumCov
from limix.core.mean import MeanKronSum
from limix.core.gp import GP2KronSum
from limix.core.gp import GP
from limix.utils.preprocess import covar_rescale
import time
import copy

if __name__=='__main__':

    # define phenotype
    N = 300
    P = 2
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

    # define kron and total covairance
    cov = SumCov(KronCov(copy.deepcopy(Cg), R), KronCov(copy.deepcopy(Cn), sp.eye(N)))

    # define gp
    gp = GP2KronSum(Y=Y, F=F, A=A, Cg=Cg, Cn=Cn, R=R)
    t0 = time.time()
    print('GP2KronSum.LML():', gp.LML())
    print('Time elapsed:', time.time() - t0)

    # compare with normal gp
    # assess compatibility with this GP
    gp0 = GP(covar = cov, mean = copy.deepcopy(gp.mean))
    t0 = time.time()
    print('GP.LML():', gp0.LML())
    print('Time elapsed:', time.time() - t0)

    t0 = time.time()
    print('GP2KronSum.LML_grad():', gp.LML_grad())
    print('Time elapsed:', time.time() - t0)

    t0 = time.time()
    print('GP.LML_grad():', gp0.LML_grad())
    print('Time elapsed:', time.time() - t0)

    params = gp.getParams()
    gp.optimize()
    gp0.optimize()

