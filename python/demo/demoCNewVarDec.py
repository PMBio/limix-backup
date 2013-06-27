import sys
sys.path.append('./../../build_mac/src/interfaces/python/')
import scipy as SP
import scipy.stats
import pdb
import h5py

import limix
import limix.modules.varianceDecomposition as VAR

# Import data
fin  = "/Users/casale/Documents/limix/limix/python/demo/data/realdata.h5py"
f    = h5py.File(fin,'r')
X    = f['X'][:]
Y    = f['Y'][:]
T    = f['T'][:]
f.close()

# Dimensions
N = (T==0).sum()
P = int(T.max()+1)
# Transform Geno and Pheno
X = SP.array(X[0:N,:],dtype=float)
Y=Y.reshape(P,N).T
Y = SP.stats.zscore(Y,0)
# Set Kronecker matrices
Kg = SP.dot(X,X.T)
Kn = SP.eye(N)

# New C++ Module
vd = limix.CNewVarianceDecomposition(Y)
vd.addFixedEffTerm("specific")
vd.addTerm("Dense",Kg)
vd.addTerm("Identity",Kg)
vd.addTerm("Dense",Kn)
vd.addTerm("Identity",Kn)
vd.initGP()

vd.getCovar().getCovariance(0).Kdim()

vd.trainGP()
min=vd.getOptimum()

# Transform Geno and Pheno and Kernels
Y1=Y.T.reshape(N*P,1)
Kg1 = Kg/Kg.diagonal().mean()
Kg1 = SP.concatenate((Kg1,Kg1),0)
Kg1 = SP.concatenate((Kg1,Kg1),1)
Kn1 = SP.concatenate((Kn,Kn),0)
Kn1 = SP.concatenate((Kn1,Kn1),1)
# Old Python/C++ Module
F  = SP.kron(SP.eye(P),SP.ones((N,1)))
C1 = limix.CTDenseCF(P)
C2 = limix.CTFixedCF(P,SP.eye(P))
C3 = limix.CTDenseCF(P)
C4 = limix.CTFixedCF(P,SP.eye(P))
C = [C1,C2,C3,C4]
K = [Kg1,Kg1,Kn1,Kn1]
vc = VAR.CVarianceDecomposition(Y1,T,F,C,K)
vc.initialise()
min0=vc.fit()

print "Params New"
print min['scales']
print "Params Old"
print min0['Params']

print "/nTimeElapsed New"
print min['time_elapsed']
print "/nTimeElapsed Old"
print min0['time_train']

