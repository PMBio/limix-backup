import sys
sys.path.append('./../../release.darwin/interfaces/python/')
import scipy as SP
import scipy.stats
import pdb
import h5py
import time
import limix

# Import data
fin  = "data/realdata.h5py"
f    = h5py.File(fin,'r')
X    = f['X'][:]
Y    = f['Y'][:]
T    = f['T'][:]
f.close()

# Dimensions
N = (T==0).sum()
P = int(T.max()+1)
# Transform Geno and Pheno
X = SP.array(X[0:N,:],dtype=float)
Y=Y.reshape(P,N).T
Y = SP.stats.zscore(Y,0)
# Set Kronecker matrices
K = SP.dot(X,X.T)
K = K/K.diagonal().mean()

print "GP KRONECKER SUM"
# Covariances
covarc1 = limix.CFreeFormCF(P)
covarc2 = limix.CFreeFormCF(P)
covarr1 = limix.CFixedCF(K)
covarr2 = limix.CFixedCF(SP.eye(N))
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
params['dataTerm']=SP.randn(1,P)
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
print "time elapsed(s): "+str(elapsed_timeFast)

paramsOpt = gpKS.getParams()
print ""
print "optimised parameters"
print SP.concatenate((paramsOpt['covarc1'],paramsOpt['covarr1'],paramsOpt['covarc2'],paramsOpt['covarr2']))
print ""

print "GP BASE"
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
Params['dataTerm']=params['dataTerm'].T
gp.setParams(Params)
# optimisation
gpopt = limix.CGPopt(gp)
start_time = time.time()
gpopt.opt()
elapsed_time = time.time() - start_time
print "time elapsed(s): "+str(elapsed_time)

print ""
print "optimised parameters"
print gp.getParams()['covar']
print ""







