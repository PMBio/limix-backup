import sys
sys.path.append('./../../build_mac/src/interfaces/python/')

import h5py
import scipy as SP
import scipy.linalg
import pylab as PL
import scipy.stats
import limix

def genPhenotype(X,T,mu,a_g,c_g,a_n,c_n):
    
    P = T.max() + 1
    N = (T==0).sum()
    S = X.shape[1]
    
    if P==2:    c[1] = c[0]
    
    Y = SP.array([])
    beta0_g = SP.randn(S)
    beta0_n = SP.randn(N)
    for p in range(P):
        
        gamma_g = SP.randn(S)
        beta_g  = a_g[p]*beta0_g+c_g[p]*gamma_g
        
        gamma_n = SP.randn(N)
        beta_n  = a_n[p]*beta0_n+c_n[p]*gamma_n
        
        Tb= (T[:,0]==p)
        x = X[Tb,:]
        y =mu[p]
        y+=SP.dot(x,beta_g)
        y+=beta_n
        Y =SP.concatenate((Y,y),0)
    
    return Y


def initialise(C1,C2,T,Y,Kpop):
    
    P = max(T)+1
    
    h1 = SP.zeros(P);
    h2 = SP.zeros(P);
    
    for p in range(P):
        It = (T[:,0]==p)
        K= Kpop[It,:][:,It]
        N= K.shape[0]
        y=Y[It]
        h1[p], h2[p]=limix.CVarianceDecomposition.aestimateHeritability(y,SP.ones(N),K)
        if h1[p]<1e-3:  h1[p]=1e-3
        if h2[p]<1e-3:  h2[p]=1e-3
            
    a     = SP.zeros(P)
    c     = SP.sqrt(h1)
    sigma = SP.sqrt(h2)

    a[0] = 1e-6

    C1.setParams(SP.concatenate((a,c)))
    C2.setParams(sigma)


# Import genotype data
N = 193
S = 1000
f = h5py.File("data/demo.h5py",'r')
X = f['geno'][:][0:N,0:S]
f.close()

# Normalize genotype data
X = X*SP.sqrt(N/((X**2).sum()))

# Set multitrait structure
P = 3
T = SP.concatenate([i*SP.ones([N,1],dtype=int) for i in xrange(P)],axis=0)
X = SP.concatenate([X for i in xrange(P)],axis=0)
    
# Set real values for parameters
mu    = 1.0*SP.ones(P)
a     = 1.0*SP.ones(P)
a[1]  = -a[1]
c     = 0.7*SP.ones(P)
sigma = 0.8*SP.ones(P)
    
# Generate phenotypes
SP.random.seed(1)
Y = genPhenotype(X,T,mu,a,c,SP.zeros(P),sigma)

# Kronecker matrix
K1 = SP.dot(X,X.T)
K2 = SP.eye(N*P)
    
# Fixed effect and trait covariances
F  = SP.kron(SP.eye(P),SP.ones((N,1)))
C1 = limix.CTLowRankCF(P)
C2 = limix.CTDiagonalCF(P)

# Initialisation of trait covariances
initialise(C1,C2,T,Y,K1)

# CVarianceDecomposition fit
vt = limix.CVarianceDecomposition()
vt.setPheno(Y)
vt.setTrait(T)
vt.setFixed(F)
vt.addCVTerm(C1,K1)
vt.addCVTerm(C2,K2)
vt.initGP()
vt.train()

# Get scales estimates
gp=vt.getGP()
ParamMask=gp.getParamMask()['covar']
Params = gp.getParams()['covar'][ParamMask==1]

# Get estmates uncertainty
H=gp.LMLhess(["covar"])
It= (ParamMask[:,0]==1)
H=H[It,:][:,It]
std = SP.sqrt(SP.linalg.inv(H).diagonal())

# Print stuff
if 1:
    print "Real Params"
    print a, c, sigma
    print "Our estimates"
    print Params
    print "Estimate uncertainity"
    print std

# Check compatibility with real values
if 0:
    comp = abs(SP.concatenate((a,c,sigma))-Params)/std
    print SP.all(comp<1.96)


