import limix
import scipy as SP
import scipy.linalg as LA
import sys
import time
sys.path.append('../')
#from linmm import lmm_fast
SP.random.seed(41)
#SP.random.seed(40)
m = 12500
n = 300
X = SP.float_(SP.random.rand(n,m) > .5)
y = SP.random.randn(n,1)
K = SP.dot(X, X.T) + SP.eye(n)*1e-4
subsample = SP.random.permutation(SP.arange(n))[:66]
delta = .2
ldelta = SP.log(delta)
S, U = LA.eigh(K)
S = SP.array(S)

UTy = SP.dot(U.T,y)
UTX = SP.dot(U.T,X)
C = SP.ones_like(UTy)
UTX_test = UTX[:,:2]
# testing of estimating weights
noderange = SP.arange(n)

def check_predictors(X, noderange, rmind):
    Xout = X[SP.ix_(noderange, rmind)]
    X_sum = SP.sum(Xout,0)
    indexes = (X_sum != X.shape[0]) & (X_sum != 0)
    print 'indexes\n', indexes, '\n\n'
    return indexes

#print "this is U now \n", U, "\n\n"
m_best, s_best, left_mean, right_mean, ll_score = limix.best_split_full_model(X, UTy, C,
                                                                    S, U,
                                                                    noderange,
                                                                    delta)

print m_best
print s_best
print left_mean
print right_mean
print ll_score
