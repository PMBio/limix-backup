# Copyright(c) 2014, The LIMIX developers (Christoph Lippert, Paolo Francesco Casale, Oliver Stegle)
# All rights reserved.
#
# LIMIX is provided under a 2-clause BSD license.
# See license.txt for the complete license.


import sys
#sys.path.append('./../../release.linux2/interfaces/python/')
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

SP.random.seed(1)

# Dimensions
N = (T==0).sum()
P = int(T.max()+1)
# Transform Geno and Pheno
X = SP.array(X[0:N,:],dtype=float)
Y=Y.reshape(P,N).T
Y = SP.stats.zscore(Y,0)
# Set Kronecker matrices
Kg = SP.dot(X,X.T)
Kg = Kg/Kg.diagonal().mean()
Kn = SP.eye(N)

# CVarianceDecomposition initializaiton and fitting
vc = VAR.CVarianceDecomposition(Y)
vc.addMultiTraitTerm(Kg)
vc.addMultiTraitTerm(Kn)
vc.addFixedTerm(SP.ones((N,1)))
vc.setScales()
params0=vc.getScales()
print vc.fit()
print vc.getOptimum()['time_train']

"""
print "\n\nMinimum Found:"
print vc.getOptimum()
"""

print "\n\nMULTI TRAIT ANALYSIS WITH GPbase"
print "\nCovar Params:"
print vc.getScales()
print "Std errors over params:"
print SP.sqrt(vc.getLaplaceCovar().diagonal())
print "DataTerm Params:"
print vc.getFixed()
print "Variances:"
print vc.getVariances()
print "Variance Components:"
print vc.getVarComponents()
print "LML:"
print vc.getLML()
print "LMLgrad:"
print vc.getLMLgrad()
# Empirical and Estimated Matrices
print "\n\nEmpirical matrix:"
print vc.getEmpTraitCovar()
print "\n\nEstimated genetic matrix:"
print vc.getEstTraitCovar(0)
print "\n\nEstimated noise matrix:"
print vc.getEstTraitCovar(1)
print "\n\nOverall estimated matrix:"
print vc.getEstTraitCovar()


# Fast Impementation
print "\n\nMULTI TRAIT ANALYSIS WITH GPkronSum"
vc.setScales(scales=params0)
print ""
print vc.fit(fast=True)
print vc.getOptimum()['time_train']
print "\nCovar Params:"
print vc.getScales()
print "DataTerm Params:"
print vc.getFixed()
print "LML:"
print vc.getLML()
print "LMLgrad:"
print vc.getLMLgrad()

# Univariate Case
print "\n\nSINGLE TRAIT ANALYSIS"
y = Y[:,0:1]
vc1 = VAR.CVarianceDecomposition(y)
vc1.addSingleTraitTerm(Kg)
vc1.addSingleTraitTerm(Kn)
fixed = SP.ones((N,1))
vc1.addFixedTerm(fixed)
vc1.setScales()
print "Parameters Found:"
print vc1.fit()
print vc1.getScales()
print "Variances:"
print vc1.getVariances()
print "Variance Components:"
print vc1.getVarComponents()

