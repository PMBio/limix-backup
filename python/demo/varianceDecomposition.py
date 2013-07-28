import sys
sys.path.append('./../../release.darwin/interfaces/python/')
import scipy as SP
import scipy.stats
import pdb
import h5py

import limix
import limix.modules.varianceDecomposition as VAR

# Import data
fin  = "./data/realdata.h5py"
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

# CVarianceDecomposition initializaiton
vc = VAR.CVarianceDecomposition(Y)
vc.addMultiTraitTerm(Kg)
vc.addMultiTraitTerm(Kn)
fixed1 = SP.kron(SP.array([1,0]),SP.ones((N,1)))
fixed2 = SP.kron(SP.array([0,1]),SP.ones((N,1)))
vc.addFixedTerm(fixed1)
vc.addFixedTerm(fixed2)

# Random initialisation and Fitting
vc.initialise()
min=vc.fit()
print "\n\nMinimum Found:"
print min
print "\n\nParameters Found:"
print min['Params']
# Taking parameters covariance matrix through Laplace Approximation
CovParams=vc.getCovParams(min)
stderr = SP.sqrt(CovParams.diagonal())
print "\n\nParams Covariance:"
print CovParams
print "\n\nStd errors over params:"
print stderr
# Minimizing multiple times
mins=vc.fit_ntimes()
print "\n\nList of Minima found in 10 minimizations:"
print mins
# Empirical and Estimated Matrices
print "\n\nEmpirical matrix:"
print vc.getEmpTraitCovar()
print "\n\nEstimated genetic matrix:"
print vc.getEstTraitCovar(0)
print "\n\nEstimated noise matrix:"
print vc.getEstTraitCovar(1)
print "\n\nOverall estimated matrix:"
print vc.getEstTraitCovar()

# Univariate Case
y = Y[:,0:1]
vc1 = VAR.CVarianceDecomposition(y)
vc1.addSingleTraitTerm(Kg)
vc1.addSingleTraitTerm(Kn)
fixed = SP.ones((N,1))
vc1.addFixedTerm(fixed)
vc1.initialise()
min1=vc1.fit()
print "\n\nMinimum Found:"
print min1
print "\n\nParameters Found:"
print min1['Params']
