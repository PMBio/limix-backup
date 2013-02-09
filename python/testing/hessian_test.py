import scipy as SP
import scipy.linalg
import sys
sys.path.append('/Users/casale/Documents/limix/limix/build_mac/src/interfaces/python/')
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


#set random seed
SP.random.seed()


print "\nTESTING THE COVARIANCE FUNCTIONS:"

n=10
n_dim=3
X=SP.randn(n,n_dim)

# SE COVAR
C=limix.CCovSqexpARD(n_dim)
C.setX(X)
n_params=C.getNumberParams()
params=SP.exp(SP.randn(n_params))
C.setParams(params)
printhessdiff(C,n_params,"SE COVAR")

# LIN ISO COVAR
C=limix.CCovLinearISO(n_dim)
C.setX(X)
n_params=C.getNumberParams()
params=SP.exp(SP.randn(n_params))
C.setParams(params)
printhessdiff(C,n_params,"LIN ISO COVAR")

"""
# LIN ISO DELTA
C=limix.CCovLinearISODelta(n_dim)
C.setX(X)
n_params=C.getNumberParams()
params=SP.exp(SP.randn(n_params))
C.setParams(params)
printhessdiff(C,n_params,"LIN ISO DELTA COVAR")
"""
    
# LIN ARD COVAR
C=limix.CCovLinearARD(n_dim)
C.setX(X)
n_params=C.getNumberParams()
params=SP.exp(SP.randn(n_params))
C.setParams(params)
printhessdiff(C,n_params,"LIN ARD COVAR")

# FIXED COVAR
C=limix.CFixedCF(SP.ones((n,n)))
n_params=C.getNumberParams()
params=SP.exp(SP.randn(n_params))
C.setParams(params)
printhessdiff(C,n_params,"FIXED COVAR")

# EYE COVAR
EyeDim=X.shape[0]
C=limix.CEyeCF(EyeDim)
n_params=C.getNumberParams()
params=SP.exp(SP.randn(n_params))
C.setParams(params)
printhessdiff(C,n_params,"EYE COVAR")

# SUM COVAR
C=limix.CSumCF()
c1=limix.CCovSqexpARD(n_dim)
c2=limix.CCovLinearARD(n_dim)
C.addCovariance(c1);
C.addCovariance(c2);
X2=SP.concatenate((X,X),axis=1)
C.setX(X2)
n_params=C.getNumberParams()
params=SP.exp(SP.randn(n_params))
C.setParams(params)
printhessdiff(C,n_params,"SUM COVAR")

# PROD COVAR
C=limix.CProductCF()
c1=limix.CCovSqexpARD(n_dim)
c2=limix.CCovLinearARD(n_dim)
C.addCovariance(c1);
C.addCovariance(c2);
X2=SP.concatenate((X,X),axis=1)
C.setX(X2)
n_params=C.getNumberParams()
params=SP.exp(SP.randn(n_params))
C.setParams(params)
printhessdiff(C,n_params,"PROD COVAR")

# LIKELIHOOD ISO
L=limix.CLikNormalIso()
n_params=L.getNumberParams()
params=SP.exp(SP.randn(n_params))
L.setParams(params)
L.setX(X)
printhessdiff(L,n_params,"LIKELIHOOD ISO")



print "\nTESTING GP CLASS:"

#genrate toy data (x,y)
n_dimensions=2
n_samples = 30
X = SP.randn(n_samples,n_dimensions)
y = SP.dot(X,SP.randn(n_dimensions,1))
y += 0.2*SP.randn(y.shape[0],y.shape[1])

#initialize LIMX objet
#1. determine covariance function:
# Squared expontential, Gaussian kernel
covar  = limix.CCovSqexpARD(n_dimensions)
#2. likelihood: Gaussian noise
ll  = limix.CLikNormalIso()

#startin parmaeters are set random
covar_params = SP.exp(SP.random.randn(covar.getNumberParams()))
lik_params = SP.exp(SP.random.randn(ll.getNumberParams())

#create hyperparameter object
hyperparams0 = limix.CGPHyperParams()
#set fields "covar" and "lik"
hyperparams0['covar'] = covar_params
hyperparams0['lik'] = lik_params

#cretae GP object
gp=limix.CGPbase(covar,ll))
#set data
#outputs
gp.setY(y)
#inputs
gp.setX(X)
#startgin parameters
gp.setParams(hyperparams0)


#Hessian testing
H_an=gp.LMLhess(["covar","lik"])
H_num=SP.zeros_like(H_an)
for i in range(H_an.shape[0]):
    for j in range(H_an.shape[0]):
        H_num[i,j]=gp.LMLhess_num(gp,i,j)
print "\nNorm of the difference between analitical and numberical hessian for a GP:"
print SP.linalg.norm(H_an-H_num)
print "\n"


