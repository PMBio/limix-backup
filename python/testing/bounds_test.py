import sys
sys.path.append('./..')
sys.path.append('./../../build/src/interfaces/python')
import limix
import scipy as SP


# LINEAR COVARIANCE MATRIX
C = limix.CCovLinearARD(2)
"""
# NON-COMPATIBLE PARAM BOUNDS
# 1. Wrong number of components
LB=SP.array([0.])
UB=float('inf')*SP.ones(2)
C.setParamBounds(LB,UB)
# 2. Upper < Lower
LB=+1.*SP.ones(2)
UB=-1.*SP.ones(2)
C.setParamBounds(LB,UB)
"""
LB=SP.array([0.,5.])
UB=SP.array([float('inf'),10.])
C.setParamBounds(LB,UB)
print C.getParamBounds0()
print C.getParamBounds()


# MULTI COVARIANCE
C1 = limix.CFixedCF()
C2 = limix.CCovSqexpARD(1)
CP = limix.CProductCF()
CP.addCovariance(C1)
CP.addCovariance(C2)
CP.setParamBounds(5*SP.ones(3),10*SP.ones(3))
print CP.getParamBounds0()
print CP.getParamBounds()

# ALSO DEFINED IN LIKELIHOODS
L=limix.CLikNormalIso()
L.setParamBounds(SP.array([0]),SP.array([10]))
L.getParamBounds()

