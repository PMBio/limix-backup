"""test script for lmm_forest"""
import scipy as SP
import limix
#test run the interface as a test
N = 100 
S = 1000
X = SP.randn(N,S)
UTy = SP.randn(N,1)
C = SP.ones([N,1])
U = SP.eye(N)
noderange = SP.arange(N)

bla=limix.best_split_full_model(X,UTy,C,S,U,noderange)
