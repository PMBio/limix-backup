import sys
sys.path.append('./../../..')
import matplotlib
matplotlib.use('PDF')
import pylab as PL
import os
import subprocess
import pdb
import sys
import numpy as NP
import numpy.linalg as LA
from optparse import OptionParser
import time
import mtSet.pycore.modules.multiTraitSetTest as MTST
from mtSet.pycore.utils.read_utils import readBimFile
from mtSet.pycore.utils.read_utils import readCovarianceMatrixFile
from mtSet.pycore.utils.read_utils import readPhenoFile
from mtSet.pycore.utils.read_utils import readCovariatesFile 
from mtSet.pycore.utils.splitter_bed import splitGeno
import mtSet.pycore.external.limix.plink_reader as plink_reader
import scipy as SP
import warnings
import mtSet.pycore.external.limix.plink_reader as plink_reader
import scipy.sparse.linalg as SSL

def computeCovarianceMatrixPlink(plink_path,out_dir,bfile,cfile,sim_type='RRM'):
    """
    computing the covariance matrix via plink
    """
    
    print "Using plink to create covariance matrix"
    cmd = '%s --bfile %s '%(plink_path,bfile)

    if sim_type=='RRM':
        # using variance standardization
        cmd += '--make-rel square '
    else:
        raise Exception('sim_type %s is not known'%sim_type)

    cmd+= '--out %s'%(os.path.join(out_dir,'plink'))
    
    subprocess.call(cmd,shell=True)

    # move file to specified file
    if sim_type=='RRM':
        old_fn = os.path.join(out_dir, 'plink.rel')
        os.rename(old_fn,cfile+'.cov')
        
        old_fn = os.path.join(out_dir, 'plink.rel.id')
        os.rename(old_fn,cfile+'.cov.id')

    if sim_type=='IBS':
        old_fn = os.path.join(out_dir, 'plink.mibs')
        os.rename(old_fn,cfile+'.cov')

        old_fn = os.path.join(out_dir, 'plink.mibs.id')
        os.rename(old_fn,cfile+'.cov.id')

def computePCsPlink(plink_path,k,out_dir,bfile,ffile):
    """
    computing the covariance matrix via plink
    """
    
    print "Using plink to compute principal components"
    cmd = '%s --bfile %s --pca %d '%(plink_path,bfile,k)
    cmd+= '--out %s'%(os.path.join(out_dir,'plink'))
    subprocess.call(cmd,shell=True)
    plink_fn = os.path.join(out_dir, 'plink.eigenvec')
    M = SP.loadtxt(plink_fn,dtype=str)
    U = SP.array(M[:,2:],dtype=float)
    U-= U.mean(0)
    U/= U.std(0)
    SP.savetxt(ffile,U)



def computePCsPython(out_dir,k,bfile,ffile):
    """ reading in """
    RV = plink_reader.readBED(bfile,useMAFencoding=True)
    X  = RV['snps']

    """ normalizing markers """
    print 'Normalizing SNPs...'
    p_ref = X.mean(axis=0)/2.
    X -= 2*p_ref

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X /= SP.sqrt(2*p_ref*(1-p_ref))
        
    hasNan = SP.any(SP.isnan(X),axis=0)
    print '%d SNPs have a nan entry. Exluding them for computing the covariance matrix.'%hasNan.sum()
    X  = X[:,~hasNan]

    
    """ computing prinicipal components """
    U,S,Vt = SSL.svds(X,k=k)
    U -= U.mean(0)
    U /= U.std(0)
    U  = U[:,::-1]
 
    """ saving to output """
    NP.savetxt(ffile, U, delimiter='\t',fmt='%.6f')    
    
    
    
def computeCovarianceMatrixPython(out_dir,bfile,cfile,sim_type='RRM'):
    print "Using python to create covariance matrix. This might be slow. We recommend using plink instead."

    if sim_type is not 'RRM':
        raise Exception('sim_type %s is not known'%sim_type)

    """ loading data """
    data = plink_reader.readBED(bfile,useMAFencoding=True)
    iid  = data['iid']
    X = data['snps']
    N = X.shape[1]
    print '%d variants loaded.'%N
    print '%d people loaded.'%X.shape[0]
    
    """ normalizing markers """
    print 'Normalizing SNPs...'
    p_ref = X.mean(axis=0)/2.
    X -= 2*p_ref

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X /= SP.sqrt(2*p_ref*(1-p_ref))
        
    hasNan = SP.any(SP.isnan(X),axis=0)
    print '%d SNPs have a nan entry. Exluding them for computing the covariance matrix.'%hasNan.sum()

    """ computing covariance matrix """
    print 'Computing relationship matrix...'
    K = SP.dot(X[:,~hasNan],X[:,~hasNan].T)
    K/= 1.*N
    print 'Relationship matrix calculation complete'
    print 'Relationship matrix written to %s.cov.'%cfile
    print 'IDs written to %s.cov.id.'%cfile

    """ saving to output """
    NP.savetxt(cfile + '.cov', K, delimiter='\t',fmt='%.6f')
    NP.savetxt(cfile + '.cov.id', iid, delimiter=' ',fmt='%s')
    



def computePCs(plink_path,k,bfile,ffile):
    """
    compute the first k principal components

    Input:
    k            :   number of principal components
    plink_path   :   plink path
    bfile        :   binary bed file (bfile.bed, bfile.bim and bfile.fam are required)
    ffile        :   name of output file
    """
    try:
        output    = subprocess.check_output('%s --version --noweb'%plink_path,shell=True)
        use_plink = float(output.split(' ')[1][1:-3])>=1.9
    except:
        use_plink = False


    assert bfile!=None, 'Path to bed-file is missing.'
    assert os.path.exists(bfile+'.bed'), '%s.bed is missing.'%bfile
    assert os.path.exists(bfile+'.bim'), '%s.bim is missing.'%bfile
    assert os.path.exists(bfile+'.fam'), '%s.fam is missing.'%bfile

    # create dir if it does not exist
    out_dir = os.path.split(ffile)[0]
    if out_dir!='' and (not os.path.exists(out_dir)):
        os.makedirs(out_dir)

    if use_plink:
        computePCsPlink(plink_path,k,out_dir,bfile,ffile)
    else:
        computePCsPython(out_dir,k,bfile,ffile)
        
def eighCovarianceMatrix(cfile):
    
    pdb.set_trace()
    S = NP.loadtxt(cfile+'.cov.eval') #S
    U = NP.loadtxt(cfile+'.cov.evec') #U
    pcs = U[:,:k]

    pcs -= pcs.mean(axis=0)
    pcs /= pcs.std(axis=0)

    NP.savetxt(ffile,pcs,fmt='%.6f')
    


def computeCovarianceMatrix(plink_path,bfile,cfile,sim_type='RRM'):
    """
    compute similarity matrix using plink

    Input:
    plink_path   :   plink path
    bfile        :   binary bed file (bfile.bed, bfile.bim and bfile.fam are required)
    cfile        :   the covariance matrix will be written to cfile.cov and the corresponding identifiers
                         to cfile.cov.id. If not specified, the covariance matrix will be written to cfile.cov and
                         the individuals to cfile.cov.id in the current folder.
    sim_type     :   {IBS/RRM} are supported
    """
    
    try:
        output    = subprocess.check_output('%s --version --noweb'%plink_path,shell=True)
        use_plink = float(output.split(' ')[1][1:-3])>=1.9
    except:
        use_plink = False

    assert bfile!=None, 'Path to bed-file is missing.'
    assert os.path.exists(bfile+'.bed'), '%s.bed is missing.'%bfile
    assert os.path.exists(bfile+'.bim'), '%s.bim is missing.'%bfile
    assert os.path.exists(bfile+'.fam'), '%s.fam is missing.'%bfile

    # create dir if it does not exist
    out_dir = os.path.split(cfile)[0]
    if out_dir!='' and (not os.path.exists(out_dir)):
        os.makedirs(out_dir)


    if use_plink:
        computeCovarianceMatrixPlink(plink_path,out_dir,bfile,cfile,sim_type=sim_type)
    else:
        computeCovarianceMatrixPython(out_dir,bfile,cfile,sim_type=sim_type)
        
def eighCovarianceMatrix(cfile):
    """
    compute similarity matrix using plink

    Input:
    cfile        :   the covariance matrix will be read from cfile.cov while the eigenvalues and the eigenverctors will
                        be written to cfile.cov.eval and cfile.cov.evec respectively
    """
    # precompute eigenvalue decomposition
    K = NP.loadtxt(cfile+'.cov')
    K+= 1e-4*SP.eye(K.shape[0])
    S,U = LA.eigh(K); S=S[::-1]; U=U[:,::-1]
    NP.savetxt(cfile+'.cov.eval',S,fmt='%.6f')
    NP.savetxt(cfile+'.cov.evec',U,fmt='%.6f')


def fit_null(Y,S_XX,U_XX,nfile,F):
    """
    fit null model

    Y       NxP phenotype matrix
    S_XX    eigenvalues of the relatedness matrix 
    U_XX    eigen vectors of the relatedness matrix
    """
    mtSet = MTST.MultiTraitSetTest(Y,S_XX=S_XX,U_XX=U_XX,F=F)

    RV = mtSet.fitNull(cache=False)
    params = NP.array([RV['params0_g'],RV['params0_n']])
    NP.savetxt(nfile+'.p0',params)
    NP.savetxt(nfile+'.nll0',RV['NLL0'])
    NP.savetxt(nfile+'.cg0',RV['Cg'])
    NP.savetxt(nfile+'.cn0',RV['Cn'])
    #if F is not None: NP.savetxt(nfile+'.f0',RV['params_mean'])
    

def preprocess(options):

    assert options.bfile!=None, 'Please specify a bfile.'
    
    """ computing the covariance matrix """
    if options.compute_cov:
       assert options.bfile!=None, 'Please specify a bfile.'
       assert options.cfile is not None, 'Specify covariance matrix basename'
       print 'Computing covariance matrix'
       t0 = time.time()
       computeCovarianceMatrix(options.plink_path,options.bfile,options.cfile,options.sim_type)
       t1 = time.time()
       print '... finished in %s seconds'%(t1-t0)
       print 'Computing eigenvalue decomposition'
       t0 = time.time()
       eighCovarianceMatrix(options.cfile) 
       t1 = time.time()
       print '... finished in %s seconds'%(t1-t0)

    """ computing principal components """
    if options.compute_PCs>0:
       assert options.ffile is not None, 'Specify fix effects basename for saving PCs'
       t0 = time.time()
       computePCs(options.plink_path,options.compute_PCs,options.bfile,options.ffile)
       t1 = time.time()
       print '... finished in %s seconds'%(t1-t0)
       

    """ fitting the null model """
    if options.fit_null:
        if options.nfile is None:
            options.nfile = os.path.split(options.bfile)[-1]
            warnings.warn('nfile not specifed, set to %s'%options.nfile)
        print 'Fitting null model'
        assert options.pfile is not None, 'phenotype file needs to be specified'
        # read pheno
        Y = readPhenoFile(options.pfile,idx=options.trait_idx)
        # read covariance
        if options.cfile is None:
            cov = {'eval':None,'evec':None}
            warnings.warn('cfile not specifed, a one variance compoenent model will be considered')
        else:
            cov = readCovarianceMatrixFile(options.cfile,readCov=False)
            assert Y.shape[0]==cov['eval'].shape[0],  'dimension mismatch'
        # read covariates
        F = None
        if options.ffile is not None:
            F = readCovariatesFile(options.ffile)
            assert Y.shape[0]==F.shape[0], 'dimensions mismatch'
        t0 = time.time()
        fit_null(Y,cov['eval'],cov['evec'],options.nfile, F)
        t1 = time.time()
        print '.. finished in %s seconds'%(t1-t0)

    """ precomputing the windows """
    if options.precompute_windows:
        if options.wfile==None:
            options.wfile = os.path.split(options.bfile)[-1] + '.%d'%options.window_size
            warnings.warn('wfile not specifed, set to %s'%options.wfile)
        print 'Precomputing windows'
        t0 = time.time()
        pos = readBimFile(options.bfile)
        nWnds,nSnps=splitGeno(pos,size=options.window_size,out_file=options.wfile+'.wnd')
        print 'Number of variants:',pos.shape[0]
        print 'Number of windows:',nWnds
        print 'Minimum number of snps:',nSnps.min()
        print 'Maximum number of snps:',nSnps.max()
        t1 = time.time()
        print '.. finished in %s seconds'%(t1-t0)

    # plot distribution of nSnps 
    if options.plot_windows:
        print 'Plotting ditribution of number of SNPs'
        plot_file = options.wfile+'.wnd.pdf'
        plt = PL.subplot(1,1,1)
        PL.hist(nSnps,30)
        PL.xlabel('Number of SNPs')
        PL.ylabel('Number of windows')
        PL.savefig(plot_file)
