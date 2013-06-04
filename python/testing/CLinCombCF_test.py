import scipy as SP
import time
import sys
import pdb
import scipy.linalg
sys.path.append('./../../build_mac/src/interfaces/python')

import limix


# CHECKING BASIC STUFF

P=2
N=4

T = SP.concatenate([i*SP.ones([N,1]) for i in xrange(P)],axis=0)

K0=SP.array([[1,0],[0,1]])

C1=limix.CTDenseCF(P)
C2=limix.CTFixedCF(P,K0)
C3=limix.CTDiagonalCF(P)

C1.setX(T)
C2.setX(T)
C3.setX(T)

n_params1 = C1.getNumberParams()
n_params2 = C2.getNumberParams()
n_params3 = C3.getNumberParams()

C1.setParams(SP.array(SP.randn(n_params1)))
C2.setParams(SP.array(SP.randn(n_params2)))
C3.setParams(SP.array(SP.randn(n_params3)))

# LINEAR COMBINATION

C = limix.CLinCombCF()
C.addCovariance(C1)
C.addCovariance(C2)
C.addCovariance(C3)

n_params = C.getNumberParams()

coeff = SP.rand(3)
C.setCoeff(coeff)

# TEST1 K
test1 = (SP.linalg.norm(C.K()-coeff[0]*C1.K()-coeff[1]*C2.K()-coeff[2]*C3.K())<1e-6)

# TEST2 grad K
test2 = C.check_covariance_Kgrad_theta(C)

# TEST3 hess K
test3=SP.zeros((n_params,n_params))
for i in range(n_params):
    for j in range(n_params):
        test3[i,j]=SP.linalg.norm(C.Khess_param(i,j)-C.Khess_param_num(C,i,j))
test3=(test3.max()<1e-6)

print test1
print test2
print test3


