import limix
import scipy as SP
import scipy.linalg as LA
import sys
import time
sys.path.append('../')
from mixed_forest import pySplittingCore as SCP
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

'''
a = S.shape
_, beta_lmm,sigma_lmm = lmm_fast.nLLeval(ldelta, UTy[:,0], UTX_test, S, MLparams=True)
a = S.shape
beta_mf, sigma_mf = SC.ml(UTy, UTX_test, S, delta)


print beta_lmm, sigma_lmm
print beta_mf, sigma_mf
'''


def check_predictors(X, noderange, rmind):
    Xout = X[SP.ix_(noderange, rmind)]
    X_sum = SP.sum(Xout,0)
    indexes = (X_sum != X.shape[0]) & (X_sum != 0)
    print 'indexes\n', indexes, '\n\n'
    return indexes


#print "this is U now \n", U, "\n\n"
m_best, s_best, left_mean, right_mean = limix.best_split_full_model(X,
                                                                  UTy,
                                                                  C,
                                                                  S,
                                                                  U,
                                                                  noderange,
                                                                  delta)


print 'success'
print 'm_best',  m_best , 'm_best_c', c_m_best
print s_best - c_s_best
print right_mean - c_right_mean
print left_mean - c_left_mean


if 0:
    noderange = SP.arange(X.shape[0])[0:n/2]
    i_OK = check_predictors(X, noderange, rmind)
    X_in  = X[:,i_OK]

    c_m_best, c_s_best, c_left_mean, c_right_mean = SC.best_split_full_model(X_in.copy(),
                                                                  UTy.copy(),
                                                                  C.copy(),
                                                                  S.copy(),
                                                                  U.copy(),
                                                                  noderange,
                                                                  delta)


    #print "this is U now \n", U, "\n\n"
    m_best, s_best, left_mean, right_mean = SCP.best_split_full_model(X_in,
                                                                  UTy,
                                                                  C,
                                                                  S,
                                                                  U,
                                                                  noderange,
                                                                  delta)


    print 'subset test'
    print 'm_best',  m_best , 'm_best_c', c_m_best
    print s_best - c_s_best
    print right_mean - c_right_mean
    print left_mean - c_left_mean


if 0:
    noderange = SP.random.permutation(SP.arange(X.shape[0]))[0:n/2]
    print 'noderange', noderange
    i_OK = check_predictors(X, noderange, rmind)
    X_in  = X[:,i_OK]

    c_m_best, c_s_best, c_left_mean, c_right_mean = SC.best_split_full_model(X_in.copy(),
                                                                  UTy.copy(),
                                                                  C.copy(),
                                                                  S.copy(),
                                                                  U.copy(),
                                                                  noderange,
                                                                  delta)


    #print "this is U now \n", U, "\n\n"
    m_best, s_best, left_mean, right_mean = SCP.best_split_full_model(X_in,
                                                                  UTy,
                                                                  C,
                                                                  S,
                                                                  U,
                                                                  noderange,
                                                                  delta)
    print 'permutation test'
    print 'm_best',  m_best , 'm_best_c', c_m_best
    print s_best - c_s_best
    print right_mean - c_right_mean
    print left_mean - c_left_mean

if 1:
    noderange = SP.random.permutation(SP.arange(X.shape[0]))[0:n/2]
    print 'more covariates test\n'
    C1 = SP.zeros_like(C)
    C1[noderange] = 1.0
    C1[0] = 1.0
    C1[2] = 1.0
    C = SP.hstack((C,C1))

    print 'C is currently:\n', C, '\n\n'

    i_OK = check_predictors(X, noderange, rmind)
    X_in  = X[:,i_OK]


    time0=time.time()
    m_best, s_best, left_mean, right_mean, _ = SCP.best_split_full_model(X_in,
                                                                  UTy,
                                                                  C,
                                                                  S,
                                                                  U,
                                                                  noderange,
                                                                  delta)

    time1=time.time()

    UT = U.T
    X_in_T = X_in.T

    time2=time.time()

#     c_m_best, c_s_best, c_left_mean, c_right_mean = SC.best_split_full_model(X_in_T,
#                                                                 UTy,
#                                                                 C,
#                                                                 S,
#                                                                 UT,
#                                                                 noderange,
#                                                                 delta)

    time3=time.time()


    UTy = SP.array(UTy, order='F')
    X_in = SP.array(X_in, order='C')
    U = SP.array(U, order='F')
    C = SP.array(C, order='F')
    S = SP.array(S, order='F')

    time4=time.time()
    c_m_best, c_s_best, c_left_mean, c_right_mean,_ = SC.best_split_full_model(X_in,
                                                                  UTy,
                                                                  C,
                                                                  S,
                                                                  U,
                                                                  noderange,
                                                                  delta)
    time5=time.time()

    print X_in_T


    print 'python version: ', time1-time0, '\n'
    print 'cpp version: ', time3-time2, '\n'
    print 'cpp version, memory cont.', time5 - time4, '\n'
    print 'benefit (factor)', (time1-time0) / (time5 - time4), '\n'
    print 'permutation test'
    print 'm_best',  m_best , 'm_best_c', c_m_best
    print s_best - c_s_best
    print right_mean - c_right_mean
    print left_mean - c_left_mean
