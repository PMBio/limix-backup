import sys
#sys.path.insert(0, '/Users/casale/Documents/limix/limix/limix') 
sys.path.insert(0, '../..') 
import scipy as sp
import limix
import ipdb
import pylab as pl
import time

sp.random.seed(0)

if __name__=='__main__':

    # generate data
    h2 = 0.3
    N = 300; P = 2; S = 1000
    X = 1.*(sp.rand(N,S)<0.2)
    beta = sp.randn(S,P)
    Yg = sp.dot(X,beta); Yg*=sp.sqrt(h2/Yg.var(0).mean())
    Yn = sp.randn(N,P); Yn*=sp.sqrt((1-h2)/Yn.var(0).mean())
    Y  = Yg+Yn; Y-=Y.mean(0); Y/=Y.std(0)
    XX = sp.dot(X,X.T)

    # add first fixed effect
    F1 = 1.*(sp.rand(N,2)<0.2); A1 = sp.eye(P)
    # add first fixed effect
    F2 = 1.*(sp.rand(N,3)<0.2); A2 = sp.ones((1,P))

    ipdb.set_trace()

    if 1:
        #1. 2KronSumGP inference
        vc = limix.VarianceDecomposition(Y)
        vc.addFixedEffect(F=F1,A=A1)
        #vc.addFixedEffect(F=F2,A=A2)
        vc.addRandomEffect(XX,trait_covar_type='freeform')
        vc.addRandomEffect(is_noise=True,trait_covar_type='freeform')
        vc.optimize()
        print vc.getTraitCovar(0)
        ipdb.set_trace()

    if 0:
        # test basic functions
        print vc.getLML()
        print vc.getLMLgrad()
        print vc.getWeights()
        print vc.getTraitCovarFun(0).K()
        print vc.getTraitCovar(0)
        ipdb.set_trace()


    if 0:
        #2. random initialization
        vc = limix.VarianceDecomposition(Y)
        vc.addFixedEffect(F=F1,A=A1)
        #vc.addFixedEffect(F=F2,A=A2)
        vc.addRandomEffect(XX,trait_covar_type='freeform')
        vc.addRandomEffect(is_noise=True,trait_covar_type='freeform')
        for ti in range(10):
            vc.optimize(init_method='random')
            print vc.getTraitCovar(0)
        ipdb.set_trace()

    if 0:
        #3. test normal gp inference
        vc = limix.VarianceDecomposition(Y)
        vc.addFixedEffect(F=F1,A=A1)
        #vc.addFixedEffect(F=F2,A=A2)
        vc.addRandomEffect(XX,trait_covar_type='freeform')
        vc.addRandomEffect(is_noise=True,trait_covar_type='freeform')
        vc.optimize(inference='GP')
        print vc.getTraitCovar(0)
        print vc.getVarianceComps()
        ipdb.set_trace()

    if 0:
        #4. single trait 
        vc = limix.VarianceDecomposition(Y[:, 0])
        vc.addFixedEffect(F=F1)
        vc.addRandomEffect(XX)
        vc.addRandomEffect(is_noise=True)
        vc.optimize()
        print vc.getVarianceComps()
        ipdb.set_trace()

    if 0:
        #5. test normal gp inference
        #tcts = ['diag', 'lowrank', 'lowrank_id', 'lowrank_diag', 'block', 'block_id', 'block_diag']
        tcts = ['freeform', 'lowrank', 'lowrank_id', 'block', 'block_id']
        for tct in tcts: 
            print tct
            vc = limix.VarianceDecomposition(Y)
            vc.addFixedEffect(F=F1, A=A1)
            #vc.addFixedEffect(F=F2, A=A2)
            vc.addRandomEffect(sp.ones((N,N))+sp.eye(N), trait_covar_type=tct)
            vc.addRandomEffect(XX, trait_covar_type=tct)
            vc.addRandomEffect(is_noise=True, trait_covar_type='freeform')
            vc.optimize()
            print vc.getTraitCovar(0)
            print vc.getTraitCovar(1)
            print vc.getTraitCovar(2)
            ipdb.set_trace()

    if 1:
        # missing data
        Ym = Y.copy()
        Inan = sp.rand(N, P) < 0.10
        Ym[Inan] = sp.nan
        vc = limix.VarianceDecomposition(Ym)
        vc.addFixedEffect(F=F1,A=A1)
        #vc.addFixedEffect(F=F2,A=A2)
        vc.addRandomEffect(XX,trait_covar_type='freeform')
        vc.addRandomEffect(is_noise=True,trait_covar_type='freeform')
        vc.optimize()
        print vc.getTraitCovar(0)
        ipdb.set_trace()

