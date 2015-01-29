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
import ipdb
import pylab as PL

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

    Sr1,Ur1 = LA.eigh(R1)
    Sr2,Ur2 = LA.eigh(R2)

    best  = SP.zeros(100)
    naive = SP.zeros(100)
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

        def f(alpha,beta,gamma,debug=False):
            if debug:   ipdb.set_trace()
            C3 = Cn+alpha*Cfg+beta*Cbg
            Sc3,Uc3 = LA.eigh(C3)
            USi2 = Uc3*Sc3**(-0.5)
            C1 = SP.dot(USi2.T,SP.dot(Cfg,USi2))
            C2 = SP.dot(USi2.T,SP.dot(Cbg,USi2))
            Sc1,Uc1 = LA.eigh(C1)
            Sc2,Uc2 = LA.eigh(C2)
            S1 = SP.kron(Sc1,Sr1-alpha)+1-gamma
            S2 = SP.kron(Sc2,Sr2-beta)+gamma
            idx1 = SP.argsort(S1)[::-1]
            idx2 = SP.argsort(S2)
            I  = (S1[idx1[::-1]]+S2[idx2]).min()>1e-4
            rv  = SP.log(S1[idx1]+S2[idx2]).sum()
            rv += R1.shape[0]*SP.log(Sc3).sum()
            rv *= I
            return rv

        Npoints = 20
        values = SP.linspace(0,1,Npoints)
        A = SP.zeros((Npoints,Npoints))
        for ai in range(Npoints):
            for bi in range(Npoints):
                A[ai,bi] = f(values[ai],values[bi],0.5)

        S,U = LA.eigh(SP.kron(Cfg,R1)+SP.kron(Cbg,R2)+SP.kron(Cn,SP.eye(R1.shape[0])))
        logdet = SP.log(S).sum()

        ld[i] = logdet
        best[i] = A.min() 
        naive[i] = f(0,0,0.5)
            
        if 1:
            i,j = SP.where(A==A.min())
            alpha = values[i]
            beta = values[j]
            C3 = Cn+alpha*Cfg+beta*Cbg
            import pylab as PL
            PL.ion()
            PL.imshow(A)
            PL.colorbar()
            S3,U3 = LA.eigh(C3)
            print S3
            f(alpha,beta,0.5,debug=True)
            ipdb.set_trace()

    PL.ion()
    PL.subplot(2,2,1)
    PL.plot(ld,naive,'.k')
    PL.plot([ld.min(),naive.max()],[ld.min(),naive.max()],'r')
    PL.subplot(2,2,2)
    PL.plot(ld,best,'.k')
    PL.plot([ld.min(),naive.max()],[ld.min(),naive.max()],'r')
    PL.show()

    ipdb.set_trace()
