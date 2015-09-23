import sys
sys.path.insert(0, '../..')
import scipy as sp
import scipy.linalg as la
from limix.core.covar import Cov3KronSumLR
from limix.core.covar import FreeFormCov 
from limix.core.gp import GP3KronSumLR
from limix.core.gp import GP
from limix.utils.preprocess import covar_rescale
import time
import copy
import pdb
import ipdb

sp.random.seed(2)

if __name__=='__main__':

    # define region and bg terms 
    N = 500
    f = 10
    P = 3
    G = 1.*(sp.rand(N, f)<0.2)
    X = 1.*(sp.rand(N, f)<0.2)
    R = covar_rescale(sp.dot(X,X.T))
    R+= 1e-4 * sp.eye(N)

    # define col covariances
    Cg = FreeFormCov(P)
    Cn = FreeFormCov(P)
    Cg.setRandomParams()
    Cn.setRandomParams()

    # define pheno
    Y = sp.randn(N, P)

    # debug covarianec
    cov = Cov3KronSumLR(Cn = Cn, Cg = Cg, R = R, G = G, rank = 1)
    cov.setRandomParams()
    U = cov.d()[:, sp.newaxis]**(0.5) * cov.W()
    pdb.set_trace()

    UU = sp.dot(U.T, U)
    L = la.cholesky(UU).T
    Li = la.inv(L)
    M = la.cholesky(sp.dot(L.T, L) + sp.eye(L.shape[0])).T
    X = sp.dot(Li.T, sp.dot(M - sp.eye(M.shape[0]), Li))

    A = sp.eye(N*P) + sp.dot(U, U.T)
    Ah = sp.eye(N*P) + sp.dot(U, sp.dot(X, U.T))
    A1 = sp.dot(Ah, Ah.T)
    print ((A-A1)**2).mean() 
    pdb.set_trace()

    K = cov.K() 
    Kh = sp.dot(la.inv(cov.L()), cov.d()[:, sp.newaxis]**(-0.5) * Ah) 
    K1 = sp.dot(Kh, Kh.T)
    print ((K-K1)**2).mean() 
    pdb.set_trace()

    B = la.inv(X) + sp.dot(U.T, U)
    Kh_inv  = la.inv(Kh)
    Kh_inv1 = sp.dot(sp.eye(N*P) - sp.dot(sp.dot(U, la.inv(B)), U.T), cov.d()[:, sp.newaxis]**(0.5) * cov.L())
    print ((Kh_inv-Kh_inv1)**2).mean() 
    pdb.set_trace()

    M = sp.randn(N, P)
    Kh_veM1 = cov.Kh_dot_ve(M)
    Kh_veM2 = sp.dot(Kh, M.reshape((N*P, 1), order='F')).reshape((N, P), order='F')
    Kh_inv_veM1 = cov.Kh_inv_dot_ve(M)
    Kh_inv_veM2 = sp.dot(Kh_inv, M.reshape((N*P, 1), order='F')).reshape((N, P), order='F')
    print ((Kh_veM1-Kh_veM2)**2).mean()
    print ((Kh_inv_veM1-Kh_inv_veM2)**2).mean()
    pdb.set_trace()


