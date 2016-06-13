import ipdb
import sys
sys.path.insert(0,'./../../..')
from limix.core.mean import mean

import scipy as SP
import scipy.linalg as LA 
import time as TIME
import copy

if __name__ == "__main__":

    # generate data
    h2 = 0.3
    N = 1000; P = 4; S = 1000
    X = 1.*(SP.rand(N,S)<0.2)
    beta = SP.randn(S,P)
    Yg = SP.dot(X,beta); Yg*=SP.sqrt(h2/Yg.var(0).mean())
    Yn = SP.randn(N,P); Yn*=SP.sqrt((1-h2)/Yn.var(0).mean())
    Y  = Yg+Yn; Y-=Y.mean(0); Y/=Y.std(0)
    XX = SP.dot(X,X.T)
    XX/= XX.diagonal().mean()
    Xr = 1.*(SP.rand(N,10)<0.2)
    Xr*= SP.sqrt(N/(Xr**2).sum())

    # define mean term
    mean = mean(Y)
    print((mean.Y))

    # add first fixed effect
    F = 1.*(SP.rand(N,2)<0.2); A = SP.eye(P)
    mean.addFixedEffect(F=F,A=A)
    # add first fixed effect
    F = 1.*(SP.rand(N,3)<0.2); A = SP.ones((1,P))
    mean.addFixedEffect(F=F,A=A)

    # rotate stuff by row and cols
    C = SP.cov(Y.T)
    Sc,Uc = LA.eigh(C)
    Sr,Ur = LA.eigh(XX)
    d = SP.kron(Sc,Sr)
    mean.d = d
    mean.Lc = Uc.T
    mean.Lr = Ur.T
    mean.LRLdiag = Sr
    mean.LCL     = C**2

    if 1:
        # calculate stuff to see if it goes through
        print((mean.Ystar()))
        print((mean.Yhat()))
        print((mean.Xstar()))
        print((mean.Xhat()))
        print((mean.XstarT_dot(SP.randn(mean.N*mean.P,mean.n_fixed_effs))))
        print((mean.Areml()))
        print((mean.beta_hat()))
        Bhat =  mean.B_hat()
        print((mean.Zstar()))

        # test grad stuff
        print((mean.LRLdiag_Xhat_tens()))
        print((mean.LRLdiag_Yhat()))
        print((mean.Areml_grad()))
        print((mean.beta_grad()))
        print((mean.Xstar_beta_grad()))


