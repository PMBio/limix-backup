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
# CLinearMean
F1=SP.ones((N,1))
W1=SP.zeros((1,P))
A1=SP.eye(P)
mean1 = limix.CKroneckerMean()
mean1.setA(A1)
mean1.setFixedEffects(F1)
mean1.setParams(W1)
F2=SP.randn(N,1)
W2=SP.zeros((1,1))
A2=SP.ones((1,P))
mean2 = limix.CKroneckerMean()
mean2.setA(A2)
mean2.setFixedEffects(F2)
mean2.setParams(W2)
mean = limix.CSumLinear()
mean.appendTerm(mean1)
mean.appendTerm(mean2)
mean.aGetParams()
# GPkronSum
gpKS = limix.CGPkronSum(Y,covarr1,covarc1,covarr2,covarc2,limix.CLikNormalNULL(),mean)
# initalization
params = limix.CGPHyperParams()
params['covarc1']=SP.randn(covarc1.getNumberParams())
params['covarc2']=SP.randn(covarc2.getNumberParams())
params['covarr1']=SP.ones(1)
params['covarr2']=SP.ones(1)
params['dataTerm']=SP.randn(P+1,1)
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
print paramsOpt['dataTerm']
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
F = SP.concatenate((SP.kron(A1.T,F1),SP.kron(A2.T,F2)),1)
mean1.setFixedEffects(F)
gp = limix.CGPbase(covar,limix.CLikNormalNULL(),mean1)
gp.setY(Y1)
# initialisation
Params = limix.CGPHyperParams()
Params['covar']=SP.concatenate((params['covarc1'],params['covarr1'],params['covarc2'],params['covarr2']))
Params['dataTerm']=params['dataTerm']
gp.setParams(Params)
# optimisation
gpopt = limix.CGPopt(gp)
start_time = time.time()
gpopt.opt()
elapsed_time = time.time() - start_time
print "time elapsed(s): "+str(elapsed_time)

paramsOpt1 = gp.getParams()
print ""
print "optimised parameters"
print paramsOpt1['covar']
print paramsOpt1['dataTerm']
print ""







