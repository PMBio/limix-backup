import sys
sys.path.append('./../../build_mac/src/interfaces/python/')
import limix
import limix.modules.varianceDecomposition as VAR
import limix.modules.FastVDMM as fVAR

import h5py
import scipy as SP
import pylab as PL
import scipy.stats
import limix
import pdb
import time

fin  = "./data/realdata.h5py"

# Import data
f    = h5py.File(fin,'r')
X    = f['X'][:]
Y    = f['Y'][:]
T    = f['T'][:]
f.close()

# Get Dimensions
X  = SP.array(X,dtype=float)
N = (T==0).sum()
P = int(T.max()+1)

# Normalize phenos
Y[0:N]     = (Y[0:N]    -SP.mean(Y[0:N]))    /SP.std(Y[0:N])
Y[N:(2*N)] = (Y[N:(2*N)]-SP.mean(Y[N:(2*N)]))/SP.std(Y[N:(2*N)])

# Kpop and Knoise
Kpop  = SP.dot(X,X.T)
Kg    = Kpop/Kpop.diagonal().mean()
Kn    = SP.kron(SP.ones((P,P)),SP.eye(N))

# Classic Variance Decomposition
F   = SP.kron(SP.eye(P),SP.ones((N,1)))
Cg1 = limix.CTDenseCF(P)
Cg2 = limix.CTFixedCF(P,SP.eye(P))
Cn1 = limix.CTDenseCF(P)
Cn2 = limix.CTFixedCF(P,SP.eye(P))
C   = [Cg1,Cg2,Cn1,Cn2]
K   = [ Kg, Kg, Kn, Kn]
vc   = VAR.CVarianceDecomposition(Y,T,F,C,K)
mins = vc.fit_ntimes(1)
min  = mins[0]

# Fast Variance Decomposition Mixed Model
Y=Y.reshape(P,N).T
K1=Kg[0:N,:][:,0:N]
C1 = [Cg1,Cg2]
C2 = [Cn1,Cn2]
fvc = fVAR.CFastVDMM(Y,C1,C2,K1)
min1 = fvc.fit()

pdb.set_trace()
# Comparison
print "\nCLASSIC VARIANCE DECOMPOSITION"
print "lml grad"
print min['LMLgrad']
print "lml"
print min['LML']
print "params"
print min['Params']
print "time train"
print min['time_train']
print "\nFAST VARIANCE DECOMPOSITION MIXED MODEL"
print "lml grad"
print min1['LMLgrad']
print "lml"
print min1['LML']
print "params"
print min1['Params']
print "time train"
print min1['time_train']

pdb.set_trace()
S=[]
S.append(Cg1.getK0())
S.append(Cg2.getK0())
S.append(Cn1.getK0())
S.append(Cn2.getK0())
K=[Kg[0:N,:][:,0:N],SP.eye(N)]
Gamma = SP.kron(S[0]+S[1],K[0])+SP.kron(S[2]+S[3],K[1])
Sbg, Kbg = fVAR.kroneckerApprox(S,K,Gamma)
Gamma1= SP.kron(Sbg,Kbg)

PL.subplot(2,2,1)
PL.imshow(Gamma)
PL.subplot(2,2,2)
PL.imshow(Gamma1)
PL.show()

pdb.set_trace()

