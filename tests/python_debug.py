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

    if 1:
        Y[0,0] = SP.nan
        #Y[1,1] = SP.nan
    #Y[10,0] = SP.nan
    #Y[100,1] = SP.nan

    vc = VAR.CVarianceDecomposition(Y)
    vc.addMultiTraitTerm(Kg)
    vc.addMultiTraitTerm(is_noise=True)
    vc.addFixedTerm(SP.ones((N,1)))
    vc.findLocalOptimum(init_method='empCov',verbose=False)        
    params = vc.getScales()
    
