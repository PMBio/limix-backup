import sys
sys.path.insert(0,'./../debug.darwin/interfaces/python')
import limix
import scipy as SP
import data
import limix.modules.varianceDecomposition as VAR
import pdb

if __name__ == '__main__':
    dir_name = './varDecomp/varDecomp'
    D = data.load(dir_name)
    S = D['X'].shape[1]
    P = D['Y'].shape[1]
    Y = D['Y']
    X = D['X']
    Y = Y[0:10]
    X = X[0:10]

    N = Y.shape[0]
    Kg = SP.dot(X,X.T)
    Kg = Kg/Kg.diagonal().mean()
    

    K0 = SP.eye(Y.shape[0])
    X0 = SP.randn(Y.shape[0],3)
    covar1 = limix.CCovLinearISO(3)
    covar1.setParams(SP.array([1.0]))
    covar1.setX(X0)
    covar2 = limix.CFixedCF(K0) 
    covar2.setParams(SP.array([1.0]))
        
    covar=limix.CSumCF()
    covar.addCovariance(covar2)
    covar.addCovariance(covar1)

    ll  = limix.CLikNormalIso()
    gp=limix.CGPbase(covar,ll)
    
    hyperparams = limix.CGPHyperParams()
    hyperparams['covar'] = SP.array([1.0,1.0])
    hyperparams['lik'] = SP.array([0.1])
    hyperparams['X']   = X0
    
    gp.setY(Y)
    gp.setX(X0)
    gp.setParams(hyperparams)
    