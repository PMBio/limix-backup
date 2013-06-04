import scipy as SP
import time
import sys
import pdb
import scipy.linalg
sys.path.append('./../../build/src/interfaces/python')

import limix

def printhessdiff(C,n_params,s):
    temp=SP.zeros((n_params,n_params))
    for i in range(n_params):
        for j in range(n_params):
            temp[i,j]=SP.linalg.norm(C.Khess_param(i,j)-C.Khess_param_num(C,i,j))
    print "\n"
    print s
    print "------------------------------------------------------------------"
    print "max{norm(H_{i,j}^{(AN)}-H_{i,j}^{(NUM)})}_{ij} = %e" % temp.max()
    print "__________________________________________________________________"



# CHECKING BASIC STUFF

P=2
N=4

T = SP.concatenate([i*SP.ones([N,1]) for i in xrange(P)],axis=0)

C=limix.CTFreeFormCF(P)
print "\nTesting "+C.getName()
NumberParams=C.getNumberParams()
Params=SP.ones(NumberParams)
print "Params: "+str(Params)
C.setParams(Params)
print "K0"
print C.getK0()
pdb.set_trace()
for i in range(Params.shape[0]):
    print "K0grad"+str(i)
    print C.getK0grad_param(i)
pdb.set_trace()

print "Test for setParamsVarCorr"
for i in range(10):
    C.setParamsVarCorr(SP.array([1,2,2*SP.rand()-1]))
    print C.getK0()
    time.sleep(1)



C=limix.CTDenseCF(P)
print "Testing "+C.getName()
Params=SP.array([2,0.5])
print "Params: "+str(Params)
C.setParams(Params)
print "K0"
print C.getK0()
pdb.set_trace()
for i in range(Params.shape[0]):
    print "K0grad"+str(i)
    print C.getK0grad_param(i)
pdb.set_trace()

K0=SP.array([[1,0],[0,1]])
C=limix.CTFixedCF(P,K0)
print "Testing "+C.getName()
Params=SP.array([2])
print "Params: "+str(Params)
C.setParams(Params)
print "K0"
print C.getK0()
pdb.set_trace()
for i in range(Params.shape[0]):
    print "K0grad"+str(i)
    print C.getK0grad_param(i)
pdb.set_trace()

C=limix.CTDiagonalCF(P)
print "Testing "+C.getName()
Params=SP.array([1,2])
print "Params: "+str(Params)
C.setParams(Params)
print "K0"
print C.getK0()
pdb.set_trace()
for i in range(Params.shape[0]):
    print "K0grad"+str(i)
    print C.getK0grad_param(i)
pdb.set_trace()

C=limix.CTLowRankCF(P)
print "Testing "+C.getName()
Params=SP.array([1,2,2])
print "Params: "+str(Params)
C.setParams(Params)
print "K0dense, K0diagonal, K0"
print C.getK0dense()
print C.getK0diagonal()
print C.getK0()
pdb.set_trace()
for i in range(Params.shape[0]):
    print "K0grad"+str(i)
    print C.getK0grad_param(i)
pdb.set_trace()


# CHECKING GRADIENT AND HESSIAN
P=4
N=2
T = SP.concatenate([i*SP.ones([N,1]) for i in xrange(P)],axis=0)
# CTFreeForm
C=limix.CTFreeFormCF(P)
n_params=C.getNumberParams()
Params=SP.exp(SP.rand(n_params))
C.setParams(Params)
C.setX(T)
print C.check_covariance_Kgrad_theta(C)
printhessdiff(C,n_params,"CTFreeForm")
# CTDense
C=limix.CTDenseCF(P)
n_params=C.getNumberParams()
Params=SP.exp(SP.rand(n_params))
C.setParams(Params)
C.setX(T)
print C.check_covariance_Kgrad_theta(C)
printhessdiff(C,n_params,"CTDense")
# CTFixed
C=limix.CTFixedCF(P,SP.ones((P,P)))
n_params=C.getNumberParams()
Params=SP.exp(SP.rand(n_params))
C.setParams(Params)
C.setX(T)
print C.check_covariance_Kgrad_theta(C)
printhessdiff(C,n_params,"CTFixed")
# CTDiagonal
C=limix.CTDiagonalCF(P)
n_params=C.getNumberParams()
Params=SP.exp(SP.rand(n_params))
C.setParams(Params)
C.setX(T)
print C.check_covariance_Kgrad_theta(C)
printhessdiff(C,n_params,"CTDiagonal")
# CTLowRank P=4
C=limix.CTLowRankCF(P)
n_params=C.getNumberParams()
Params=SP.exp(SP.rand(n_params))
C.setParams(Params)
C.setX(T)
print C.check_covariance_Kgrad_theta(C)
printhessdiff(C,n_params,"CTLowRank_P=4")
# CTLowRank P=2
P=2
N=2
T = SP.concatenate([i*SP.ones([N,1]) for i in xrange(P)],axis=0)
C=limix.CTLowRankCF(P)
n_params=C.getNumberParams()
Params=SP.exp(SP.rand(n_params))
C.setParams(Params)
C.setX(T)
print C.check_covariance_Kgrad_theta(C)
printhessdiff(C,n_params,"CTLowRank_P=2")


