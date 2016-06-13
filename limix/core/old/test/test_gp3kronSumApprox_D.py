import sys
import scipy as SP
import scipy.linalg as LA
sys.path.insert(0,'/Users/casale/Documents/limix/limix')
from limix.core.covar import freeform
from limix.core.gp.gp3kronSumApprox import gp3kronSumApprox 
import limix.core.optimize.optimize_bfgs as OPT
sys.path.append('./../../../build/release.darwin/interfaces/python/limix/modules')
import varianceDecomposition as VAR
import ipdb
import h5py
import pylab as PL
PL.ion()

def genPheno(G=None,X=None,var_g=None,var_x=None,P=None):
    var_n = 1-var_g-var_x
    Yg = SP.dot(G,SP.randn(G.shape[1],P))
    Yx = SP.dot(X,SP.randn(X.shape[1],P))
    Yn = SP.randn(X.shape[0],P)
    Yg*= SP.sqrt(var_g/Yg.var(0).mean())
    Yx*= SP.sqrt(var_x/Yx.var(0).mean())
    Yn*= SP.sqrt(var_n/Yn.var(0).mean())
    RV = Yg+Yx+Yn
    RV-= RV.mean(0); RV/= RV.std(0)
    return RV 

if __name__=='__main__':

    P = 4

    seed = int(sys.argv[1])
    SP.random.seed(seed)

    # import data
    fname = 'data/arab107_preprocessed.hdf5'
    f = h5py.File(fname,'r')
    X = f['genotype'][:]
    X-= X.mean(0); X/=X.std(0)
    G = X[:,0:200]

    Y = genPheno(G=G,X=X,var_g=0.10,var_x=0.40,P=P)
    XX = SP.dot(X,X.T)
    XX/= XX.diagonal().mean()
    XX+= 1e-4*SP.eye(XX.shape[0])
    GG = SP.dot(G,G.T)
    GG/= GG.diagonal().mean()
    GG+= 1e-4*SP.eye(GG.shape[0])

    C1 = freeform(P)
    C2 = freeform(P)
    Cn = freeform(P)
    gp = gp3kronSumApprox(Y=Y,C1=C1,C2=C2,Cn=Cn,R1=GG,R2=XX,tol=1e-16) 

    if 0:
        Kinit = SP.cov(Y.T)/3 
        C1.setCovariance(Kinit)
        C2.setCovariance(Kinit)
        Cn.setCovariance(Kinit)
    else:
        C1.setRandomParams()
        C2.setRandomParams()
        Cn.setRandomParams()
    params = gp.getParams()
    gp.setParams(params)

    gp.checkGradient()

    if 0:
        gp.K.optimizeAB(n=100)
        print(('a:', gp.K.a))
        print(('b:', gp.K.b))

        ipdb.set_trace()
        gp.K.optimizeABgrad()

    for i in range(10):

        ipdb.set_trace()

        conv,info = OPT.opt_hyper(gp,params,factr=1e-3)
        print(conv)

        print('C1')
        print((C1.K()))
        print('C2')
        print((C2.K()))
        print('Cn')
        print((Cn.K()))

        ipdb.set_trace()
        print(('before a and b opt:', gp.LML()))
        gp.K.optimizeAB(n=100)
        print(('after a and b opt:', gp.LML()))
        print(('a:', gp.K.a))
        print(('b:', gp.K.b))
        params = gp.getParams()



