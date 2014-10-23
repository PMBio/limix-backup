import scipy as SP
# Create some random data
# and simulate some simple linear model effect
import limix
n = 10
m = 100
X = SP.random.randn(n, m)
y = SP.random.randn(n, 1)
C = SP.ones_like(y)
S = SP.ones_like(y)
U = SP.identity(n)
print U

noderange = SP.arange(n)
delta = .5
# model simple linear effect
beta = .5
y += X[:, 0:1] * beta
print 'done'
mBest = 0
print type(mBest)
sBest = 0.
left_mean = 0.
right_mean = 0.
ll_score = 0.
limix.best_split_full_model(mBest, sBest, left_mean, right_mean, ll_score, X, y,
                            C, S, U, noderange, delta)
