import sys
sys.path.append('./../../..')
import copy

from mtSet.pycore.mean import mean
import mtSet.pycore.gp.gp2kronSum as gp2kronSum
import mtSet.pycore.optimize.optimize_bfgs as OPT
import mtSet.pycore.covariance as covariance
import scipy as SP
import scipy.linalg as LA
import h5py
import pdb
import pylab as PL
import copy
import os
import cPickle

def fitSingleTraitModel(Y,XX=None,S_XX=None,U_XX=None,verbose=False):
    """ fit single trait model """
    N,P = Y.shape
    RV = {}
    Cg = covariance.lowrank(1)
    Cn = covariance.lowrank(1)
    gp = gp2kronSum(mean(Y[:,0:1]),Cg,Cn,XX=XX,S_XX=S_XX,U_XX=U_XX)
    params0 = {'Cg':SP.sqrt(0.5)*SP.ones(1),'Cn':SP.sqrt(0.5)*SP.ones(1)}
    var = SP.zeros((P,2))
    conv1 = SP.zeros(P,dtype=bool)
    for p in range(P):
        if verbose:
            print '.. fitting variance trait %d'%p
        gp.setY(Y[:,p:p+1])
        conv1[p],info = OPT.opt_hyper(gp,params0,factr=1e3)
        var[p,0] = Cg.K()[0,0]
        var[p,1] = Cn.K()[0,0]
    RV['conv1'] = conv1
    RV['varST'] = var
    return RV

def fitPairwiseModel(Y,XX=None,S_XX=None,U_XX=None,verbose=False):
    N,P = Y.shape
    """ initilizes parameters """
    RV = fitSingleTraitModel(Y,XX=XX,S_XX=S_XX,U_XX=U_XX,verbose=verbose)
    Cg = covariance.freeform(2)
    Cn = covariance.freeform(2)
    gp = gp2kronSum(mean(Y[:,0:2]),Cg,Cn,XX=XX,S_XX=S_XX,U_XX=U_XX)
    conv2 = SP.ones((P,P),dtype=bool)
    rho_g = SP.ones((P,P))
    rho_n = SP.ones((P,P))
    for p1 in range(P):
        for p2 in range(p1):
            if verbose:
                print '.. fitting correlation (%d,%d)'%(p1,p2)
            gp.setY(Y[:,[p1,p2]])
            Cg_params0 = SP.array([SP.sqrt(RV['varST'][p1,0]),1e-6*SP.randn(),SP.sqrt(RV['varST'][p2,0])])
            Cn_params0 = SP.array([SP.sqrt(RV['varST'][p1,1]),1e-6*SP.randn(),SP.sqrt(RV['varST'][p2,1])])
            params0 = {'Cg':Cg_params0,'Cn':Cn_params0}
            conv2[p1,p2],info = OPT.opt_hyper(gp,params0,factr=1e3)
            rho_g[p1,p2] = Cg.K()[0,1]/SP.sqrt(Cg.K().diagonal().prod())
            rho_n[p1,p2] = Cn.K()[0,1]/SP.sqrt(Cn.K().diagonal().prod())
            conv2[p2,p1] = conv2[p1,p2]; rho_g[p2,p1] = rho_g[p1,p2]; rho_n[p2,p1] = rho_n[p1,p2]
    RV['Cg0'] = rho_g*SP.dot(SP.sqrt(RV['varST'][:,0:1]),SP.sqrt(RV['varST'][:,0:1].T))
    RV['Cn0'] = rho_n*SP.dot(SP.sqrt(RV['varST'][:,1:2]),SP.sqrt(RV['varST'][:,1:2].T))
    RV['conv2'] = conv2
    #3. regularizes covariance matrices
    offset_g = abs(SP.minimum(LA.eigh(RV['Cg0'])[0].min(),0))+1e-4
    offset_n = abs(SP.minimum(LA.eigh(RV['Cn0'])[0].min(),0))+1e-4
    RV['Cg0_reg'] = RV['Cg0']+offset_g*SP.eye(P)
    RV['Cn0_reg'] = RV['Cn0']+offset_n*SP.eye(P)
    RV['params0_Cg']=LA.cholesky(RV['Cg0_reg'])[SP.tril_indices(P)]
    RV['params0_Cn']=LA.cholesky(RV['Cn0_reg'])[SP.tril_indices(P)]
    return RV

