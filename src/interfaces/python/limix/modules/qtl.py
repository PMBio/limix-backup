"""@package docstring
qtl.py contains wrappers around C++ Limix objects to streamline common tasks in GWAS.
"""

import scipy as SP
import limix
import limix.utils.preprocess as preprocess
import time
import pdb


def simple_lmm(snps,pheno,K=None,covs=None):
    """
    Univariate fixed effects linear mixed model test for all SNPs
    ----------------------------------------------------------------------------
    Input:
    snps   [N x S] SP.array of S SNPs for N individuals
    pheno  [N x 1] SP.array of 1 phenotype for N individuals
    K      [N x N] SP.array of LMM-covariance/kinship koefficients (optional)
                   If not provided, then linear regression analysis is performed
    covs   [N x D] SP.array of D covariates for N individuals
    -----------------------------------------------------------------------------
    Output:
    lmix LMM object
    """
    t0=time.time()
    if K is None:
        K=SP.eye(N)
    lm = limix.CLMM()
    lm.setK(K)
    lm.setSNPs(snps)
    lm.setPheno(pheno)
    if covs is None:
        covs = SP.ones((snps.shape[0],1))
    lm.setCovs(covs)
    lm.process()
    t1=time.time()
    print ("finished GWAS testing in %.2f seconds" %(t1-t0))
    return lm


def interact_GxG(pheno,snps1,snps2=None,K=None,covs=None):
    """
    Epistasis test between two sets of SNPs
    ----------------------------------------------------------------------------
    Input:
    snps1  [N x S1] SP.array of S1 SNPs for N individuals
    snps2  [N x S2] SP.array of S2 SNPs for N individuals
    pheno  [N x 1] SP.array of 1 phenotype for N individuals
    K      [N x N] SP.array of LMM-covariance/kinship koefficients (optional)
                   If not provided, then linear regression analysis is performed
    covs   [N x D] SP.array of D covariates for N individuals
    -----------------------------------------------------------------------------
    Output:
    pv     [S2 x S1] SP.array of P values for epistasis tests beten all SNPs in 
           snps1 and snps2
    """
    if K is None:
        K=SP.eye(N)
    N=snps1.shape[0]
    if snps2 is None:
        snps2 = snps1
    return interact_GxE(snps=snps1,pheno=pheno,env=snps2,covs=covs,K=K)


def interact_GxE_1dof(snps,pheno,env,K=None,covs=None):
    """
    Univariate GxE fixed effects interaction linear mixed model test for all 
    pairs of SNPs and environmental variables.
    ----------------------------------------------------------------------------
    Input:
    snps   [N x S] SP.array of S SNPs for N individuals
    pheno  [N x 1] SP.array of 1 phenotype for N individuals
    env    [N x E] SP.array of E environmental variables for N individuals
    K      [N x N] SP.array of LMM-covariance/kinship koefficients (optional)
                   If not provided, then linear regression analysis is performed
    covs   [N x D] SP.array of D covariates for N individuals
    -----------------------------------------------------------------------------
    Output:
    pv     [E x S] SP.array of P values for interaction tests between all 
           E environmental variables and all S SNPs
    """
    N=snps.shape[0]
    if K is None:
        K=SP.eye(N)
    if covs is None:
        covs = SP.ones((N,1))
    assert (env.shape[0]==N and pheno.shape[0]==N and K.shape[0]==N and K.shape[1]==N and covs.shape[0]==N), "shapes missmatch"
    Inter0 = SP.ones((N,1))
    pv = SP.zeros((env.shape[1],snps.shape[1]))
    print ("starting %i interaction scans for %i SNPs each." % (env.shape[1], snps.shape[1]))
    t0=time.time()
    for i in xrange(env.shape[1]):
        t0_i = time.time()
        cov_i = SP.concatenate((covs,env[:,i:(i+1)]),1)
        lm_i = simple_interaction(snps=snps,pheno=pheno,covs=cov_i,Inter=env[:,i:(i+1)],Inter0=Inter0)
        pv[i,:]=lm_i.getPv()[0,:]
        t1_i = time.time()
        print ("Finished %i out of %i interaction scans in %.2f seconds."%((i+1),env.shape[1],(t1_i-t0_i)))
    t1 = time.time()
    print ("-----------------------------------------------------------\nFinished all %i interaction scans in %.2f seconds."%(env.shape[1],(t1-t0)))
    return pv
        

def phenSpecificEffects(snps,pheno1,pheno2,K=None,covs=None):
    """
    Univariate fixed effects interaction test for phenotype specific SNP effects
    ----------------------------------------------------------------------------
    Input:
    snps   [N x S] SP.array of S SNPs for N individuals (test SNPs)
    pheno1 [N x 1] SP.array of 1 phenotype for N individuals
    pheno2 [N x 1] SP.array of 1 phenotype for N individuals
    K      [N x N] SP.array of LMM-covariance/kinship koefficients (optional)
                   If not provided, then linear regression analysis is performed
    covs   [N x D] SP.array of D covariates for N individuals
    -----------------------------------------------------------------------------
    Output:
    lmix LMM object
    """
    N=snps.shape[0]
    if K is None:
        K=SP.eye(N)
    assert (pheno1.shape[1]==pheno2.shape[1]), "Only consider equal number of phenotype dimensions"
    if covs is None:
        covs = SP.ones(N,1)
    assert (pheno1.shape[1]==1 and pheno2.shape[1]==1 and pheno1.shape[0]==N and pheno2.shape[0]==N and K.shape[0]==N and K.shape[1]==N and covs.shape[0]==N), "shapes missmatch"
    Inter = SP.zeros((N*2,1))
    Inter[0:N,0]=1
    Inter0 = SP.ones((N*2,1))
    Yinter=SP.concatenate((pheno1,pheno2),0)
    Xinter = SP.tile(snps,(2,1))
    Covitner= SP.tile(covs(2,1))
    lm = simple_interaction(snps=Xinter,pheno=Yinter,covs=Covinter,Inter=Inter,Inter0=Inter0)
    return lm

def simple_interaction(snps,pheno,Inter,covs = None,K=None,Inter0=None):
    """
    I-variate fixed effects interaction test for phenotype specific SNP effects
    ----------------------------------------------------------------------------
    Input:
    snps   [N x S] SP.array of S SNPs for N individuals (test SNPs)
    pheno  [N x 1] SP.array of 1 phenotype for N individuals
    Inter  [N x I] SP.array of I interaction variables to be tested for N 
                   individuals (optional)
                   If not provided, only the SNP is included in the null model
    Inter0 [N x I0] SP.array of I0 interaction variables to be included in the 
                    background model when testing for interaction with Inter
    K      [N x N] SP.array of LMM-covariance/kinship koefficients (optional)
                   If not provided, then linear regression analysis is performed
    covs   [N x D] SP.array of D covariates for N individuals
    -----------------------------------------------------------------------------
    Output:
    lmix LMM object
    """
    N=snps.shape[0]
    if covs is None:
        covs = SP.ones((N,1))
    if K is None:
        K = SP.eye(N)
    if Inter0 is None:
        Inter0=SP.ones([N,1])
    assert (pheno.shape[0]==N and K.shape[0]==N and K.shape[1]==N and covs.shape[0]==N and Inter0.shape[0]==N and Inter.shape[0]==N), "shapes missmatch"
    lmi = limix.CInteractLMM()
    lmi.setK(K)
    lmi.setSNPs(snps)
    lmi.setPheno(pheno)
    lmi.setCovs(covs)
    lmi.setInter0(Inter0)
    lmi.setInter(Inter)
    lmi.process()
    return lmi

def forward_lmm(snps,pheno,K=None,covs=None,threshold = 5e-8, maxiter = 2):
    """
    univariate fixed effects test with forward selection
    ----------------------------------------------------------------------------
    Input:
    snps   [N x S] SP.array of S SNPs for N individuals (test SNPs)
    pheno  [N x 1] SP.array of 1 phenotype for N individuals
    K      [N x N] SP.array of LMM-covariance/kinship koefficients (optional)
                   If not provided, then linear regression analysis is performed
    covs   [N x D] SP.array of D covariates for N individuals
    threshold      (float) P-value thrashold for inclusion in forward selection
                   (default 5e-8)
    maxiter        (int) maximum number of interaction scans. First scan is
                   without inclusion, so maxiter-1 inclusions can be performed.
                   (default 2)
    -----------------------------------------------------------------------------
    Output:
    lm             lmix LMM object
    iadded         array of indices of SNPs included in order of inclusion
    pvadded        array of Pvalues obtained by the included SNPs in iteration
                   before inclusion
    pvall   [maxiter x S] SP.array of Pvalues for all iterations
    """
    t0=time.time()
    if K is None:
        K=SP.eye(N)
    lm = limix.CLMM()
    lm.setK(K)
    lm.setSNPs(snps)
    lm.setPheno(pheno)
    iadded = []
    pvadded = []
    if covs is None:
        covs = SP.ones([snps.shape[0],1])
    lm.setCovs(covs)
    lm.process()
    pvall = SP.zeros((maxiter,snps.shape[1]))
    t1=time.time()
    print ("finished GWAS testing in %.2f seconds" %(t1-t0))
    pv = lm.getPv()
    pvall[0:1,:]=pv
    imin= pv.argmin()
    niter = 1
    while (pv[0,imin]<threshold) and niter<maxiter:
        t0=time.time()
        iadded.append(imin)
        covs=SP.concatenate((covs,snps[:,imin:(imin+1)]),1)
        lm.setCovs(covs)
        lm.process()
        pv = lm.getPv()
        pvall[niter:niter+1,:]=pv
        imin= pv.argmin()
        pvadded.append(pv[0,imin])
        t1=time.time()
        print ("finished GWAS testing in %.2f seconds" %(t1-t0))
        niter=niter+1
    return lm,iadded,pvadded,pvall
