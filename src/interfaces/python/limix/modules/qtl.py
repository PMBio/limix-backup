"""@package docstring
qtl.py contains wrappers around C++ Limix objects to streamline common tasks in GWAS.
"""

import scipy as SP
import limix
import limix.utils.preprocess as preprocess
import limix.modules.varianceDecomposition as VAR
import time
import pdb



## KroneckerLMM functions

def kronecker_lmm(snps,phenos,Asnps=None,K1r=None,K2r=None,K1c=None,K2c=None,covs=None,Acovs=None):
    """
    simple wrapepr for kroneckerLMM code
    """
    #0. checks dimensions
    N  = phenos.shape[0]
    P  = phenos.shape[1]
    
    if K1r==None:
        K1r = SP.dot(snps,snps.T)
    else:
        assert K1r.shape[0]==N, 'K1r: dimensions dismatch'
        assert K1r.shape[1]==N, 'K1r: dimensions dismatch'

    if K2r==None:
        K2r = SP.eye(N)
    else:
        assert K2r.shape[0]==N, 'K2r: dimensions dismatch'
        assert K2r.shape[1]==N, 'K2r: dimensions dismatch'

    #1. run GP model to infer suitable covariance structure
    if K1c==None or K2c==None:
        print ".. Training the backgrond covariance with a GP model"
        vc = VAR.CVarianceDecomposition(phenos)
        vc.addMultiTraitTerm(XX,covar_type='rank1_diag')
        vc.addMultiTraitTerm(Kn,covar_type='rank1_diag')
        fixed = SP.ones([N,1])
        vc.addFixedTerm(fixed)
        vc.setScales()
        pdb.set_trace()
        start = time.time()
        conv = vc.fit(fast=True)
        K1c = vc.getEstTraitCovar(0)
        K2c = vc.getEstTraitCovar(1)
        time_el = time.time()-start
        print "Bg model trained in %.2f s" % time_el
    else:
        assert K1c.shape[0]==P, 'K1c: dimensions dismatch'
        assert K1c.shape[1]==P, 'K1c: dimensions dismatch'
        assert K2c.shape[0]==P, 'K2c: dimensions dismatch'
        assert K2c.shape[1]==P, 'K2c: dimensions dismatch'
    
    #2. run kroneckerLMM
    if covs==None:
        covs = SP.ones([N,1])#was covs = SP.zeros([N,1])
    Xcov = covs
    if Acovs is None:
        Acovs = SP.eye(P)
    if Asnps is None:
        Asnps = SP.ones([1,P])
    lmm = limix.CKroneckerLMM()
    lmm.setK1r(K1r)
    lmm.setK1c(K1c)
    lmm.setK2r(K2r)
    lmm.setK2c(K2c)
    lmm.setSNPs(snps)
    #add covariates
    lmm.addCovariates(Xcov,Acovs)
    #add SNP design
    lmm.setSNPcoldesign(Asnps)
    lmm.setPheno(phenos)
    lmm.setNumIntervalsAlt(0)
    lmm.setNumIntervals0(100)
    lmm.process()
    return lmm


def simple_lmm(snps,pheno,K=None,covs=None,numIntervals0=None,numIntervalsAlt=None):
    """
    Univariate fixed effects linear mixed model test for all SNPs
    ----------------------------------------------------------------------------
    Input:
    snps   [N x S] SP.array of S SNPs for N individuals
    pheno  [N x 1] SP.array of 1 phenotype for N individuals
    K      [N x N] SP.array of LMM-covariance/kinship koefficients (optional)
                   If not provided, then linear regression analysis is performed
    covs   [N x D] SP.array of D covariates for N individuals
    numIntervals0: number of bins for delta optimization (0 model)
    numIntervalsAlt: number of bins for delta optimization (alt. model)
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
    if numIntervals0 is not None:
        lm.setNumIntervals0(numIntervals0)
    if numIntervalsAlt is not None:
        lm.setNumIntervalsAlt(numIntervalsAlt)
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


def forward_lmm_kronecker(snps,phenos,Asnps = None,K1r=None,K1c=None,K2r=None,K2c=None,covs=None,Acovs=None,threshold = 5e-8, maxiter = 2):
    """
    kronecker fixed effects test with forward selection
    ----------------------------------------------------------------------------
    Input:
    snps   [N x S] SP.array of S SNPs for N individuals (test SNPs)
    pheno  [N x P] SP.array of 1 phenotype for N individuals
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
    P=phenos.shape[1]
    t0=time.time()
    if Asnps is None:
        Asnps = SP.ones((1,P))
    lm = kronecker_lmm(snps=snps,phenos=phenos,Asnps=Asnps,K1r=K1r,K2r=K2r,K1c=K1c,K2c=K2c,covs=covs,Acovs=Acovs)
    #get pv
    #start stuff
    iadded = []
    pvadded = []
    pvall = SP.zeros((maxiter,snps.shape[1]))
    t1=time.time()
    print ("finished GWAS testing in %.2f seconds" %(t1-t0))
    pv = lm.getPv()
    pvall[0:1,:]=pv
    imin= pv.argmin()
    niter = 1
    while (pv[0,imin]<threshold) and niter<maxiter:
        t0=time.time()
        pvadded.append(pv[0,imin])
        iadded.append(imin)
        #covs=SP.concatenate((covs,snps[:,imin:(imin+1)]),1)
        lm.addCovariates(snps[:,imin:(imin+1)],Asnps)
        lm.process()
        pv = lm.getPv()
        pvall[niter:niter+1,:]=pv
        imin= pv.argmin()
        t1=time.time()
        print ("finished GWAS testing in %.2f seconds" %(t1-t0))
        niter=niter+1
    return lm,iadded,pvadded,pvall

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
        pvadded.append(pv[0,imin])
        covs=SP.concatenate((covs,snps[:,imin:(imin+1)]),1)
        lm.setCovs(covs)
        lm.process()
        pv = lm.getPv()
        pvall[niter:niter+1,:]=pv
        imin= pv.argmin()
        t1=time.time()
        print ("finished GWAS testing in %.2f seconds" %(t1-t0))
        niter=niter+1
    return lm,iadded,pvadded,pvall
