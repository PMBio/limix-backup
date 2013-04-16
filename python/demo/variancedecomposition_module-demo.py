import sys
sys.path.append('./../../build_mac/src/interfaces/python')

import h5py
import scipy as SP
import scipy.linalg
import pylab as PL
import scipy.stats
import limix
import limix.modules.varianceDecomposition as VAR

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
    
    Y=Y[:,SP.newaxis]
    
    return Y


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

# Fixed effect
F  = SP.kron(SP.eye(P),SP.ones((N,1)))
# Trait Matrices
C1 = limix.CTDenseCF(P)
C2 = limix.CTFixedCF(P,SP.eye(P))
C3 = limix.CTDiagonalCF(P)
# Kronecker matrix
K1 = SP.dot(X,X.T)
K2 = K1
K3 = SP.kron(SP.ones((P,P)),SP.eye(N))
# Variance Decomposition Design
C = [C1,C2,C3]
K = [K1,K2,K3]

# CVarianceDecomposition fit
vc = VAR.CVarianceDecomposition(Y,T,F,C,K)

# Random initialisation
vc.initialise()

# Fitting
min=vc.fit()
# Taking parameters covariance matrix through Laplace Approximation
CovParams=vc.getCovParams(min)
# Minimizing multiple times
mins=vc.fit_ntimes()

# Test heritabilities
print "\nTest Estimate Heritabilities"
print vc.estimateHeritabilities(K1)

print "\nResults Single Minimization"
print "Params0"
print min["Params0"]
print "LML0"
print min["LML0"]
print "Params"
print min["Params"]
print "LML"
print min["LML"]
print "LMLgrad"
print min["LMLgrad"]
print "TraitCovar"
print min["TraitCovar"]
print "EmpiricalTraitCovar"
print vc.getEmpTraitCov()
print "CovParams"
print CovParams

