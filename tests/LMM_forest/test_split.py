import scipy as SP
# Create some random data
# and simulate some simple linear model effect
n = 10
m = 100
X = SP.random.randn(n, m)
y = SP.random.randn(n, 1)
beta = .5
y += X[:, 0:1] * beta
print 'done'
