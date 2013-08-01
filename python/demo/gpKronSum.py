import sys
sys.path.append('./../../release.darwin/interfaces/python/')
import scipy as SP
import scipy.stats
import pdb
import h5py
import time
import limix

def genGeno(N,S):
    X  = (SP.rand(N,S)<0.2)*1.
    return X

def genPheno(X,P):
    N = X.shape[0]
    dp = SP.ones(P); dp[1]=-1
    Y = SP.zeros((N,P))
    gamma0_g = SP.randn(S)
    gamma0_n = SP.randn(N)
    for p in range(P):
        gamma_g = SP.randn(S)
        gamma_n = SP.randn(N)
        beta_g  = dp[p]*gamma0_g+gamma_g
        beta_n  = gamma0_n+gamma_n
        y=SP.dot(X,beta_g)
        y+=beta_n
        Y[:,p]=y
    Y=SP.stats.zscore(Y,0)
    return Y

# Dimensions
P = 10
N = 10
S = 1e5

# Generate Data
X = genGeno(N,S)
Y = genPheno(X,P)
Kg = SP.dot(X,X.T)
Kg = Kg/Kg.diagonal().mean()
Kn = SP.eye(N)

# Covariances
covarc1 = limix.CFreeFormCF(P)
covarc2 = limix.CFreeFormCF(P)
covarr1 = limix.CFixedCF(Kg)
covarr2 = limix.CFixedCF(Kn)
covarr1.setParamMask(SP.zeros(1))
covarr2.setParamMask(SP.zeros(1))
# CKroneckerMean
F=SP.ones((N,1))
A=SP.eye(P)
W=SP.zeros((1,P))
mean = limix.CLinearMean()
mean = limix.CKroneckerMean()
mean.setA(A)
mean.setFixedEffects(F)
mean.setParams(W)
# GPkronSum
gpKS = limix.CGPkronSum(Y,covarr1,covarc1,covarr2,covarc2,limix.CLikNormalNULL(),mean)
# initalization
params = limix.CGPHyperParams()
params['covarc1']=SP.randn(covarc1.getNumberParams())
params['covarc2']=SP.randn(covarc2.getNumberParams())
params['covarr1']=SP.ones(1)
params['covarr2']=SP.ones(1)
params['dataTerm']=SP.zeros((1,P))
gpKS.setParams(params)
# Paramask
paramMask = limix.CGPHyperParams()
paramMask['covarc1']=covarc1.getParamMask()
paramMask['covarc2']=covarc2.getParamMask()
paramMask['covarr1']=covarr1.getParamMask()
paramMask['covarr2']=covarr2.getParamMask()
# optimisation
gpKSopt = limix.CGPopt(gpKS)
start_time = time.time()
gpKSopt.opt()
elapsed_timeFast = time.time() - start_time

# NormalGP
Y1=Y.T.reshape(N*P,1)
# Covars
covar = limix.CSumCF()
covar.addCovariance(limix.CKroneckerCF(covarc1,covarr1))
covar.addCovariance(limix.CKroneckerCF(covarc2,covarr2))
# Mean
mean1 = limix.CLinearMean()
mean1.setFixedEffects(SP.kron(A,F))
gp = limix.CGPbase(covar,limix.CLikNormalNULL(),mean1)
gp.setY(Y1)
# initialisation
Params = limix.CGPHyperParams()
Params['covar']=SP.concatenate((params['covarc1'],params['covarr1'],params['covarc2'],params['covarr2']))
Params['dataTerm']=SP.zeros((P,1))
gp.setParams(Params)
# optimisation
gpopt = limix.CGPopt(gp)
start_time = time.time()
gpopt.opt()
elapsed_time = time.time() - start_time

print "GP kronecker Sum"
params=gpKS.getParams()
print SP.concatenate((params['covarc1'],params['covarr1'],params['covarc2'],params['covarr2']))
print elapsed_time
print "GP base"
print gp.getParams()['covar']
print elapsed_timeFast