import sys
import scipy 
import pdb
import h5py
from utils import *

import sys
sys.path.insert(0,'../../')
import limix.utils.utils
from settings import *
import scipy as SP
import scipy.linalg as LA
import scipy.linalg as LA
import scipy.sparse.linalg as SLA
import ipdb
import pylab as PL

def compute_pcs(C,R,nPCs=10):

    N = R[0].shape[0]
    P = C[0].shape[0]
    n_terms = len(R)

    def _Kx(x):
        """
        compute K x
        """
        X = SP.reshape(x,(N,P),order='F')
        B = SP.zeros((N,P))
        for i in range(n_terms):
            B += SP.dot(R[i],SP.dot(X,C[i]))
        b = SP.reshape(B,(N*P),order='F')
        return b
        
    linop = SLA.LinearOperator((N*P,N*P),matvec=_Kx,dtype='float64')
    S,U = SLA.eigsh(linop,k=nPCs)

    X = U*((S**0.5)[SP.newaxis,:])

    return X

def lowrank_diag(C,R,nPCs=10):

    N = R[0].shape[0]
    P = C[0].shape[0]
    n_terms = len(R)

    X = compute_pcs(C,R,nPCs=nPCs)

    full_diag = SP.zeros(N*P)
    for i in range(n_terms):
        full_diag += SP.kron(C[i].diagonal(),R[i].diagonal())

    lr_diag = (X**2).sum(1)

    d = full_diag-lr_diag

    return X,d


if __name__ == "__main__":
        
    fname = './../core/tests/data/arab107_preprocessed.hdf5'
    f = h5py.File(fname,'r')
    X  = f['genotype'][0:100,:]
    X1 = X[:,0:200]
    X2 = X[:]
    R1 = SP.dot(X1,X1.T)
    R1/= R1.diagonal().mean()
    R1+= 1e-4*SP.eye(R1.shape[0])
    R2 = SP.dot(X2,X2.T)
    R2/= R2.diagonal().mean()
    R2+= 1e-4*SP.eye(R2.shape[0])
    R = [R1,R2,SP.eye(R1.shape[0])]

    ld1 = SP.zeros(100)
    ld    = SP.zeros(100)
    for i in range(100):
        print i

        # simulate trait-trait covariance matrices
        P  = 4
        weights = scipy.random.rand(3)
        weights/= weights.sum()
        Cfg = weights[0]*sim_psd_matrix(N=P,n_dim=1,jitter=1e-4)
        Cbg = weights[1]*sim_psd_matrix(N=P,n_dim=1,jitter=1e-4)
        Cn  = weights[2]*sim_psd_matrix(N=P,n_dim=1,jitter=1e-4)

        C = [Cfg,Cbg,Cn]

        X,d = lowrank_diag(C,R)

        B = SP.eye(X.shape[1])+SP.dot(X.T,d[:,SP.newaxis]*X)
        Sb,Ub = LA.eigh(B)
        ld1[i] = SP.log(d).sum()+SP.log(Sb).sum()

        N = R[0].shape[0]
        P = C[0].shape[0]
        n_terms = len(R)
        K = SP.zeros(N*P)
        for i in range(n_terms):    K = SP.kron(C[i],R[i])
        Sk,Uk = LA.eigh(K)
        ld[i] = SP.log(Sk).sum()


    PL.plot(ld,ld1,'.')
    PL.show()

