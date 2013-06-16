import sys
sys.path.append('./../../build/src/interfaces/python')
import limix

import scipy as SP
import scipy.linalg
import pdb


C1 = limix.CTFixedCF(2,SP.ones((2,2)))
C2 = limix.CTFixedCF(3,SP.floor(10*SP.rand(3,3)))

C = limix.CKroneckerCF()
C.setRowCovariance(C1)
C.setColCovariance(C2)

C.setXr(SP.array([0,1]))
C.setXc(SP.array([0,1,2]))

n_params = C.getNumberParams()
C.setParams(SP.randn(n_params))


# Test K
print ((C.K()-SP.kron(C1.K(),C2.K())).max()<1e-6)

# Test Kgrad
print C.check_covariance_Kgrad_theta(C)

# Test Khess
test=SP.zeros((n_params,n_params))
for i in range(n_params):
    for j in range(n_params):
        test[i,j]=SP.linalg.norm(C.Khess_param(i,j)-C.Khess_param_num(C,i,j))
print (test.max()<1e-6)