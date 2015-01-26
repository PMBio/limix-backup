import sys
import scipy 
import pdb
import h5py
from utils import *

import sys
sys.path.insert(0,'../../')
import limix.core.covar.cov3kronSum as cov3kronSum
import limix.core.covar.freeform as freeform
import limix.core.gp.gp3kronSumApprox as gp3kronSumApprox
import limix.core.optimize.optimize_bfgs as optimize_bfgs
import limix.utils.utils
from settings import *

def opt_gp3kronSumApprox(Y,R1,R2):
    # optimizing
    P = Y.shape[1]
    cov1 = freeform(P);
    cov2 = freeform(P);
    cov3 = freeform(P);
    gp = gp3kronSumApprox.gp3kronSumApprox(Y=Y,Cr=cov1,Cg=cov2,Cn=cov3,XX=R1,GG=R2,tol=1E-3,bound='up')
    cov1.setRandomParams()
    cov2.setRandomParams()
    cov3.setRandomParams()
    params = gp.getParams()
    gp.setParams(params)
    conv,info = optimize_bfgs.opt_hyper(gp,params,factr=1e3)

    RV = {}
    RV['LML'] = scipy.array([gp.LML_debug()])
    RV['LB']  = scipy.array([gp.LML()])
    gp.setBound('low')
    RV['UB']        = scipy.array([gp.LML()])
    RV['C1']        = cov1.K()
    RV['C2']        = cov2.K()
    RV['C3']        = cov3.K()
    
    return RV

def set_gp3kronSumAProx(Y,R1,R2,C1,C2,C3):
    P = Y.shape[1]
    cov1 = freeform(P);
    cov2 = freeform(P);
    cov3 = freeform(P);
    gp = gp3kronSumApprox.gp3kronSumApprox(Y=Y,Cr=cov1,Cg=cov2,Cn=cov3,XX=R1,GG=R2,tol=1E-3,bound='up')
    cov1.setCovariance(C1)
    cov2.setCovariance(C2)
    cov3.setCovariance(C3)
    params = gp.getParams()
    gp.setParams(params)

    
    RV = {}
    RV['LML'] = scipy.array([gp.LML_debug()])
    RV['LB']  = scipy.array([gp.LML()])
    gp.setBound('low')
    RV['UB']        = scipy.array([gp.LML()])
    RV['C1']        = cov1.K()
    RV['C2']        = cov2.K()
    RV['C3']        = cov3.K()

    return RV

if __name__ == "__main__":
    # notation:
    # K1: foreground covariance
    # K2: background covariance
    # K3: noise covariance
    
    nReps = 100
    dataset_name = sys.argv[1]
    seed         = int(sys.argv[2])
    scipy.random.seed(seed)
    
    if dataset_name=='arab':
        fn = CFG['arab']['data']
        X,chrom,pos = load_arabidopsis(fn,debug=False)
        N = X.shape[0]
        I = scipy.eye(N)
        
        R1 = scipy.dot(X[:,chrom==3],X[:,chrom==3].T)
        R1 /= scipy.diag(R1).mean()
        R1 += 1e-2*I
        
        R2 = scipy.dot(X[:,chrom!=3],X[:,chrom!=3].T)
        R2/= scipy.diag(R2).mean()
        R2+= 1e-2*I
    elif dataset_name=='NFBC':
        f = h5py.File(CFG['nfbc']['data'],'r')
        R2 = f['Kpop'][:1000][:,:1000]
        I  = scipy.eye(R2.shape[0])
        R2+= 1e-2*I
        
        chrom = f['chrom'][:]
        X = f['X'][:1000][:,chrom==15]
        R1 = scipy.dot(X,X.T)
        R1/= scipy.diag(R1).mean()
        R1+= 1e-2*I
        f.close()

        
    # simulate trait-trait covariance matrices
    P  = 4
    N  = R1.shape[0]
    I  = scipy.eye(N)
    weights = scipy.random.rand(3)
    weights/= weights.sum()
    C1 = weights[0]*sim_psd_matrix(N=P,n_dim=P,jitter=0.1)
    C2 = weights[1]*sim_psd_matrix(N=P,n_dim=P,jitter=0.1)
    C3 = weights[2]*sim_psd_matrix(N=P,n_dim=P,jitter=0.1)

    # simulate phenotypes
    Y1 = sim_kronecker(C1,R1)
    Y2 = sim_kronecker(C2,R2)
    Y3 = sim_kronecker(C3,I)
    Y  = Y1 + Y2 + Y3

    
    # optimizing
    RV = {}
    tmp = opt_gp3kronSumApprox(Y,R1,R2)
    RV['OPT_K3_eps0'] = {'LML':tmp['LML'],'LB':tmp['LB'], 'UB':tmp['UB'], 'Cr':tmp['C1'], 'Cg':tmp['C2'], 'Cn':tmp['C3']}
    tmp = set_gp3kronSumAProx(Y,R1,R2,C1,C2,C3)
    RV['TRUE_K3_eps0'] = {'LML':tmp['LML'],'LB':tmp['LB'], 'UB':tmp['UB'], 'Cr':tmp['C1'], 'Cg':tmp['C2'], 'Cn':tmp['C3']}
    tmp = opt_gp3kronSumApprox(Y,R2,R1)
    RV['OPT_K3_eps1'] = {'LML':tmp['LML'],'LB':tmp['LB'], 'UB':tmp['UB'], 'Cr':tmp['C2'], 'Cg':tmp['C1'], 'Cn':tmp['C3']}
    tmp = set_gp3kronSumAProx(Y,R2,R1,C2,C1,C3)
    RV['TRUE_K3_eps1'] = {'LML':tmp['LML'],'LB':tmp['LB'], 'UB':tmp['UB'], 'Cr':tmp['C2'], 'Cg':tmp['C1'], 'Cn':tmp['C3']}
    
    # writing out
    fn = 'out/approximation/%s_%d.hdf'%(dataset_name,seed)
    f  = h5py.File(fn,'w')
    for key in RV.keys():
        group = f.create_group(key)
        limix.utils.utils.dumpDictHdf5(RV[key],group)
    f.close()
        
        
