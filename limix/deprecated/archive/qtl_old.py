"""
qtl.py contains wrappers around C++ Limix objects to streamline common tasks in GWAS.
"""

import scipy as SP
import scipy.stats as ST
import limix
import limix.utils.preprocess as preprocess
import limix.deprecated.modules.varianceDecomposition as VAR
import limix.utils.fdr as FDR
import time

#TODO: externally visible function?
#I propose to make this internal using _
def estimateKronCovariances(phenos,K1r=None,K1c=None,K2r=None,K2c=None,covs=None,Acovs=None,covar_type='lowrank_diag',rank=1):
    """
    estimates the background covariance model before testing

    Args:
        phenos: [N x P] SP.array of P phenotypes for N individuals
        K1r:    [N x N] SP.array of LMM-covariance/kinship koefficients (optional)
                        If not provided, then linear regression analysis is performed
        K1c:    [P x P] SP.array of LMM-covariance/kinship koefficients (optional)
                        If not provided, then linear regression analysis is performed
        K2r:    [N x N] SP.array of LMM-covariance/kinship koefficients (optional)
                        If not provided, then linear regression analysis is performed
        K2c:    [P x P] SP.array of LMM-covariance/kinship koefficients (optional)
                        If not provided, then linear regression analysis is performed
        covs:           list of SP.arrays holding covariates. Each covs[i] has one corresponding Acovs[i]
        Acovs:          list of SP.arrays holding the phenotype design matrices for covariates.
                        Each covs[i] has one corresponding Acovs[i].
        covar_type:     type of covaraince to use. Default 'freeform'. possible values are
                        'freeform': free form optimization,
                        'fixed': use a fixed matrix specified in covar_K0,
                        'diag': optimize a diagonal matrix,
                        'lowrank': optimize a low rank matrix. The rank of the lowrank part is specified in the variable rank,
                        'lowrank_id': optimize a low rank matrix plus the weight of a constant diagonal matrix. The rank of the lowrank part is specified in the variable rank,
                        'lowrank_diag': optimize a low rank matrix plus a free diagonal matrix. The rank of the lowrank part is specified in the variable rank,
                        'block': optimize the weight of a constant P x P block matrix of ones,
                        'block_id': optimize the weight of a constant P x P block matrix of ones plus the weight of a constant diagonal matrix,
                        'block_diag': optimize the weight of a constant P x P block matrix of ones plus a free diagonal matrix,
        rank:           rank of a possible lowrank component (default 1)

    Returns:
        CVarianceDecomposition object
    """
    print(".. Training the backgrond covariance with a GP model")
    vc = VAR.CVarianceDecomposition(phenos)
    if K1r is not None:
        vc.addRandomEffect(K1r,covar_type=covar_type,rank=rank)
    if K2r is not None:
        #TODO: fix this; forces second term to be the noise covariance
        vc.addRandomEffect(is_noise=True,K=K2r,covar_type=covar_type,rank=rank)
    for ic  in range(len(Acovs)):
        vc.addFixedEffect(covs[ic],Acovs[ic])
    start = time.time()
    conv = vc.findLocalOptimum(fast=True)
    assert conv, "CVariance Decomposition has not converged"
    time_el = time.time()-start
    print(("Background model trained in %.2f s" % time_el))
    return vc

#TODO: externally visible function?
#what does this do?
def updateKronCovs(covs,Acovs,N,P):
    """
    make sure that covs and Acovs are lists
    """
    if (covs is None) and (Acovs is None):
        covs = [SP.ones([N,1])]
        Acovs = [SP.eye(P)]

    if Acovs is None or covs is None:
        raise Exception("Either Acovs or covs is None, while the other isn't")

    if (type(Acovs)!=list) and (type(covs)!=list):
        Acovs= [Acovs]
        covs = [covs]
    if (type(covs)!=list) or (type(Acovs)!=list) or (len(covs)!=len(Acovs)):
        raise Exception("Either Acovs or covs is not a list or they missmatch in length")
    return covs, Acovs

def simple_interaction_kronecker_deprecated(snps,phenos,covs=None,Acovs=None,Asnps1=None,Asnps0=None,K1r=None,K1c=None,K2r=None,K2c=None,covar_type='lowrank_diag',rank=1,searchDelta=False):
    """
    I-variate fixed effects interaction test for phenotype specific SNP effects.
    (Runs multiple likelihood ratio tests and computes the P-values in python from the likelihood ratios)

    Args:
        snps:   [N x S] SP.array of S SNPs for N individuals (test SNPs)
        phenos: [N x P] SP.array of P phenotypes for N individuals
        covs:           list of SP.arrays holding covariates. Each covs[i] has one corresponding Acovs[i]
        Acovs:          list of SP.arrays holding the phenotype design matrices for covariates.
                        Each covs[i] has one corresponding Acovs[i].
        Asnps1:         list of SP.arrays of I interaction variables to be tested for N
                        individuals. Note that it is assumed that Asnps0 is already included.
                        If not provided, the alternative model will be the independent model
        Asnps0:         single SP.array of I0 interaction variables to be included in the
                        background model when testing for interaction with Inters
        K1r:    [N x N] SP.array of LMM-covariance/kinship koefficients (optional)
                        If not provided, then linear regression analysis is performed
        K1c:    [P x P] SP.array of LMM-covariance/kinship koefficients (optional)
                        If not provided, then linear regression analysis is performed
        K2r:    [N x N] SP.array of LMM-covariance/kinship koefficients (optional)
                        If not provided, then linear regression analysis is performed
        K2c:    [P x P] SP.array of LMM-covariance/kinship koefficients (optional)
                        If not provided, then linear regression analysis is performed
        covar_type:     type of covaraince to use. Default 'freeform'. possible values are
                        'freeform': free form optimization,
                        'fixed': use a fixed matrix specified in covar_K0,
                        'diag': optimize a diagonal matrix,
                        'lowrank': optimize a low rank matrix. The rank of the lowrank part is specified in the variable rank,
                        'lowrank_id': optimize a low rank matrix plus the weight of a constant diagonal matrix. The rank of the lowrank part is specified in the variable rank,
                        'lowrank_diag': optimize a low rank matrix plus a free diagonal matrix. The rank of the lowrank part is specified in the variable rank,
                        'block': optimize the weight of a constant P x P block matrix of ones,
                        'block_id': optimize the weight of a constant P x P block matrix of ones plus the weight of a constant diagonal matrix,
                        'block_diag': optimize the weight of a constant P x P block matrix of ones plus a free diagonal matrix,
        rank:           rank of a possible lowrank component (default 1)
        searchDelta:    Boolean indicator if delta is optimized during SNP testing (default False)

    Returns:
        pv:     P-values of the interaction test
        lrt0:   log likelihood ratio statistics of the null model
        pv0:    P-values of the null model
        lrt:    log likelihood ratio statistics of the interaction test
        lrtAlt: log likelihood ratio statistics of the alternative model
        pvAlt:  P-values of the alternative model
    """
    S=snps.shape[1]
    #0. checks
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

    covs,Acovs = updateKronCovs(covs,Acovs,N,P)

    #Asnps can be several designs
    if (Asnps0 is None):
        Asnps0 = [SP.ones([1,P])]
    if Asnps1 is None:
        Asnps1 = [SP.eye([P])]
    if (type(Asnps0)!=list):
        Asnps0 = [Asnps0]
    if (type(Asnps1)!=list):
        Asnps1 = [Asnps1]
    assert (len(Asnps0)==1) and (len(Asnps1)>0), "need at least one Snp design matrix for null and alt model"

    #one row per column design matrix
    pv = SP.zeros((len(Asnps1),snps.shape[1]))
    lrt = SP.zeros((len(Asnps1),snps.shape[1]))
    pvAlt = SP.zeros((len(Asnps1),snps.shape[1]))
    lrtAlt = SP.zeros((len(Asnps1),snps.shape[1]))

    #1. run GP model to infer suitable covariance structure
    if K1c==None or K2c==None:
        vc = estimateKronCovariances(phenos=phenos, K1r=K1r, K2r=K2r, K1c=K1c, K2c=K2c, covs=covs, Acovs=Acovs, covar_type=covar_type, rank=rank)
        K1c = vc.getEstTraitCovar(0)
        K2c = vc.getEstTraitCovar(1)
    else:
        assert K1c.shape[0]==P, 'K1c: dimensions dismatch'
        assert K1c.shape[1]==P, 'K1c: dimensions dismatch'
        assert K2c.shape[0]==P, 'K2c: dimensions dismatch'
        assert K2c.shape[1]==P, 'K2c: dimensions dismatch'

    #2. run kroneckerLMM for null model
    lmm = limix.CKroneckerLMM()
    lmm.setK1r(K1r)
    lmm.setK1c(K1c)
    lmm.setK2r(K2r)
    lmm.setK2c(K2c)
    lmm.setSNPs(snps)
    #add covariates
    for ic  in range(len(Acovs)):
        lmm.addCovariates(covs[ic],Acovs[ic])
    lmm.setPheno(phenos)
    if searchDelta:      lmm.setNumIntervalsAlt(100)
    else:                   lmm.setNumIntervalsAlt(0)
    lmm.setNumIntervals0(100)
    #add SNP design
    lmm.setSNPcoldesign(Asnps0[0])
    lmm.process()
    dof0 = Asnps0[0].shape[0]
    pv0 = lmm.getPv()
    lrt0 = ST.chi2.isf(pv0,dof0)
    for iA in range(len(Asnps1)):
        dof1 = Asnps1[iA].shape[0]
        dof = dof1-dof0
        lmm.setSNPcoldesign(Asnps1[iA])
        lmm.process()
        pvAlt[iA,:] = lmm.getPv()[0]
        lrtAlt[iA,:] = ST.chi2.isf(pvAlt[iA,:],dof1)
        lrt[iA,:] = lrtAlt[iA,:] - lrt0[0] # Don't need the likelihood ratios, as null model is the same between the two models
        pv[iA,:] = ST.chi2.sf(lrt[iA,:],dof)
    return pv,lrt0,pv0,lrt,lrtAlt,pvAlt

#TODO: (O.S), I have changed the parametrization of delta optimization steps. Happy with that?
#TODO: Do we really want to keep these "simple_XXX" names? Which functions are simple, which ones are not? I don't like it.
def simple_interaction_kronecker(snps,phenos,covs=None,Acovs=None,Asnps1=None,Asnps0=None,K1r=None,K1c=None,K2r=None,K2c=None,covar_type='lowrank_diag',rank=1,NumIntervalsDelta0=100,NumIntervalsDeltaAlt=0,searchDelta=False):
    """
    I-variate fixed effects interaction test for phenotype specific SNP effects

    Args:
        snps:   [N x S] SP.array of S SNPs for N individuals (test SNPs)
        phenos: [N x P] SP.array of P phenotypes for N individuals
        covs:           list of SP.arrays holding covariates. Each covs[i] has one corresponding Acovs[i]
        Acovs:          list of SP.arrays holding the phenotype design matrices for covariates.
                        Each covs[i] has one corresponding Acovs[i].
        Asnps1:         list of SP.arrays of I interaction variables to be tested for N
                        individuals. Note that it is assumed that Asnps0 is already included.
                        If not provided, the alternative model will be the independent model
        Asnps0:         single SP.array of I0 interaction variables to be included in the
                        background model when testing for interaction with Inters
        K1r:    [N x N] SP.array of LMM-covariance/kinship koefficients (optional)
                        If not provided, then linear regression analysis is performed
        K1c:    [P x P] SP.array of LMM-covariance/kinship koefficients (optional)
                        If not provided, then linear regression analysis is performed
        K2r:    [N x N] SP.array of LMM-covariance/kinship koefficients (optional)
                        If not provided, then linear regression analysis is performed
        K2c:    [P x P] SP.array of LMM-covariance/kinship koefficients (optional)
                        If not provided, then linear regression analysis is performed
        covar_type:     type of covaraince to use. Default 'freeform'. possible values are
                        'freeform': free form optimization,
                        'fixed': use a fixed matrix specified in covar_K0,
                        'diag': optimize a diagonal matrix,
                        'lowrank': optimize a low rank matrix. The rank of the lowrank part is specified in the variable rank,
                        'lowrank_id': optimize a low rank matrix plus the weight of a constant diagonal matrix. The rank of the lowrank part is specified in the variable rank,
                        'lowrank_diag': optimize a low rank matrix plus a free diagonal matrix. The rank of the lowrank part is specified in the variable rank,
                        'block': optimize the weight of a constant P x P block matrix of ones,
                        'block_id': optimize the weight of a constant P x P block matrix of ones plus the weight of a constant diagonal matrix,
                        'block_diag': optimize the weight of a constant P x P block matrix of ones plus a free diagonal matrix,
        rank:           rank of a possible lowrank component (default 1)
        NumIntervalsDelta0:  number of steps for delta optimization on the null model (100)
        NumIntervalsDeltaAlt:number of steps for delta optimization on the alt. model (0 - no optimization)
        searchDelta:     Carry out delta optimization on the alternative model? if yes We use NumIntervalsDeltaAlt steps
    Returns:
        pv:     P-values of the interaction test
        pv0:    P-values of the null model
        pvAlt:  P-values of the alternative model
    """
    S=snps.shape[1]
    #0. checks
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

    covs,Acovs = updateKronCovs(covs,Acovs,N,P)

    #Asnps can be several designs
    if (Asnps0 is None):
        Asnps0 = [SP.ones([1,P])]
    if Asnps1 is None:
        Asnps1 = [SP.eye([P])]
    if (type(Asnps0)!=list):
        Asnps0 = [Asnps0]
    if (type(Asnps1)!=list):
        Asnps1 = [Asnps1]
    assert (len(Asnps0)==1) and (len(Asnps1)>0), "need at least one Snp design matrix for null and alt model"

    #one row per column design matrix
    pv = SP.zeros((len(Asnps1),snps.shape[1]))
    lrt = SP.zeros((len(Asnps1),snps.shape[1]))
    pvAlt = SP.zeros((len(Asnps1),snps.shape[1]))
    lrtAlt = SP.zeros((len(Asnps1),snps.shape[1]))

    #1. run GP model to infer suitable covariance structure
    if K1c==None or K2c==None:
        vc = estimateKronCovariances(phenos=phenos, K1r=K1r, K2r=K2r, K1c=K1c, K2c=K2c, covs=covs, Acovs=Acovs, covar_type=covar_type, rank=rank)
        K1c = vc.getEstTraitCovar(0)
        K2c = vc.getEstTraitCovar(1)
    else:
        assert K1c.shape[0]==P, 'K1c: dimensions dismatch'
        assert K1c.shape[1]==P, 'K1c: dimensions dismatch'
        assert K2c.shape[0]==P, 'K2c: dimensions dismatch'
        assert K2c.shape[1]==P, 'K2c: dimensions dismatch'

    #2. run kroneckerLMM for null model
    lmm = limix.CKroneckerLMM()
    lmm.setK1r(K1r)
    lmm.setK1c(K1c)
    lmm.setK2r(K2r)
    lmm.setK2c(K2c)
    lmm.setSNPs(snps)
    #add covariates
    for ic  in range(len(Acovs)):
        lmm.addCovariates(covs[ic],Acovs[ic])
    lmm.setPheno(phenos)

    #delta serch on alt. model?
    if searchDelta:
        lmm.setNumIntervalsAlt(NumIntervalsDeltaAlt)
        lmm.setNumIntervals0_inter(NumIntervalsDeltaAlt)
    else:
        lmm.setNumIntervalsAlt(0)
        lmm.setNumIntervals0_inter(0)


    lmm.setNumIntervals0(NumIntervalsDelta0)
    #add SNP design
    lmm.setSNPcoldesign0_inter(Asnps0[0])
    for iA in range(len(Asnps1)):
        lmm.setSNPcoldesign(Asnps1[iA])
        lmm.process()

        pvAlt[iA,:] = lmm.getPv()[0]
        pv[iA,:] = lmm.getPv()[1]
        pv0 = lmm.getPv()[2]
    return pv,pv0,pvAlt

## KroneckerLMM functions

def kronecker_lmm(snps,phenos,covs=None,Acovs=None,Asnps=None,K1r=None,K1c=None,K2r=None,K2c=None,covar_type='lowrank_diag',rank=1,NumIntervalsDelta0=100,NumIntervalsDeltaAlt=0,searchDelta=False):
    """
    simple wrapper for kroneckerLMM code

    Args:
        snps:   [N x S] SP.array of S SNPs for N individuals (test SNPs)
        phenos: [N x P] SP.array of P phenotypes for N individuals
        covs:           list of SP.arrays holding covariates. Each covs[i] has one corresponding Acovs[i]
        Acovs:          list of SP.arrays holding the phenotype design matrices for covariates.
                        Each covs[i] has one corresponding Acovs[i].
        Asnps:          single SP.array of I0 interaction variables to be included in the
                        background model when testing for interaction with Inters
                        If not provided, the alternative model will be the independent model
        K1r:    [N x N] SP.array of LMM-covariance/kinship koefficients (optional)
                        If not provided, then linear regression analysis is performed
        K1c:    [P x P] SP.array of LMM-covariance/kinship koefficients (optional)
                        If not provided, then linear regression analysis is performed
        K2r:    [N x N] SP.array of LMM-covariance/kinship koefficients (optional)
                        If not provided, then linear regression analysis is performed
        K2c:    [P x P] SP.array of LMM-covariance/kinship koefficients (optional)
                        If not provided, then linear regression analysis is performed
        covar_type:     type of covaraince to use. Default 'freeform'. possible values are
                        'freeform': free form optimization,
                        'fixed': use a fixed matrix specified in covar_K0,
                        'diag': optimize a diagonal matrix,
                        'lowrank': optimize a low rank matrix. The rank of the lowrank part is specified in the variable rank,
                        'lowrank_id': optimize a low rank matrix plus the weight of a constant diagonal matrix. The rank of the lowrank part is specified in the variable rank,
                        'lowrank_diag': optimize a low rank matrix plus a free diagonal matrix. The rank of the lowrank part is specified in the variable rank,
                        'block': optimize the weight of a constant P x P block matrix of ones,
                        'block_id': optimize the weight of a constant P x P block matrix of ones plus the weight of a constant diagonal matrix,
                        'block_diag': optimize the weight of a constant P x P block matrix of ones plus a free diagonal matrix,
        rank:           rank of a possible lowrank component (default 1)
        NumIntervalsDelta0:  number of steps for delta optimization on the null model (100)
        NumIntervalsDeltaAlt:number of steps for delta optimization on the alt. model (0 - no optimization)
        searchDelta:    Boolean indicator if delta is optimized during SNP testing (default False)

    Returns:
        CKroneckerLMM object
        P-values for all SNPs from liklelihood ratio test
    """
    #0. checks
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

    covs,Acovs = updateKronCovs(covs,Acovs,N,P)

    #Asnps can be several designs
    if Asnps is None:
        Asnps = [SP.ones([1,P])]
    if (type(Asnps)!=list):
        Asnps = [Asnps]
    assert len(Asnps)>0, "need at least one Snp design matrix"

    #one row per column design matrix
    pv = SP.zeros((len(Asnps),snps.shape[1]))

    #1. run GP model to infer suitable covariance structure
    if K1c==None or K2c==None:
        vc = estimateKronCovariances(phenos=phenos, K1r=K1r, K2r=K2r, K1c=K1c, K2c=K2c, covs=covs, Acovs=Acovs, covar_type=covar_type, rank=rank)
        K1c = vc.getEstTraitCovar(0)
        K2c = vc.getEstTraitCovar(1)
    else:
        assert K1c.shape[0]==P, 'K1c: dimensions dismatch'
        assert K1c.shape[1]==P, 'K1c: dimensions dismatch'
        assert K2c.shape[0]==P, 'K2c: dimensions dismatch'
        assert K2c.shape[1]==P, 'K2c: dimensions dismatch'

    #2. run kroneckerLMM

    lmm = limix.CKroneckerLMM()
    lmm.setK1r(K1r)
    lmm.setK1c(K1c)
    lmm.setK2r(K2r)
    lmm.setK2c(K2c)
    lmm.setSNPs(snps)
    #add covariates
    for ic  in range(len(Acovs)):
        lmm.addCovariates(covs[ic],Acovs[ic])
    lmm.setPheno(phenos)


    #delta serch on alt. model?
    if searchDelta:
        lmm.setNumIntervalsAlt(NumIntervalsDeltaAlt)
    else:
        lmm.setNumIntervalsAlt(0)
    lmm.setNumIntervals0(NumIntervalsDelta0)

    for iA in range(len(Asnps)):
        #add SNP design
        lmm.setSNPcoldesign(Asnps[iA])
        lmm.process()
        pv[iA,:] = lmm.getPv()[0]
    return lmm,pv


def simple_lmm(snps,pheno,K=None,covs=None, test='lrt',NumIntervalsDelta0=100,NumIntervalsDeltaAlt=0,searchDelta=False):
    """
    Univariate fixed effects linear mixed model test for all SNPs

    Args:
        snps:   [N x S] SP.array of S SNPs for N individuals
        pheno:  [N x 1] SP.array of 1 phenotype for N individuals
        K:      [N x N] SP.array of LMM-covariance/kinship koefficients (optional)
                        If not provided, then linear regression analysis is performed
        covs:   [N x D] SP.array of D covariates for N individuals
        test:   'lrt' for likelihood ratio test (default) or 'f' for F-test
        NumIntervalsDelta0:  number of steps for delta optimization on the null model (100)
        NumIntervalsDeltaAlt:number of steps for delta optimization on the alt. model (0 - no optimization)
        searchDelta:     Carry out delta optimization on the alternative model? if yes We use NumIntervalsDeltaAlt steps

    Returns:
        limix LMM object
    """
    t0=time.time()
    if K is None:
        K=SP.eye(snps.shape[0])
    lm = limix.CLMM()
    lm.setK(K)
    lm.setSNPs(snps)
    lm.setPheno(pheno)
    if covs is None:
        covs = SP.ones((snps.shape[0],1))
    lm.setCovs(covs)
    if test=='lrt':
        lm.setTestStatistics(0)
    elif test=='f':
        lm.setTestStatistics(1)
    else:
        print(test)
        raise NotImplementedError("only f or lrt are implemented")
    #set number of delta grid optimizations?
    lm.setNumIntervals0(NumIntervalsDelta0)
    if searchDelta:
        lm.setNumIntervalsAlt(NumIntervalsDeltaAlt)
    else:
        lm.setNumIntervalsAlt(0)
    lm.process()
    t1=time.time()
    print(("finished GWAS testing in %.2f seconds" %(t1-t0)))
    return lm

#TODO: we need to fix. THis does not work as interact_GxE is not existing
#I vote we also use **kw_args to forward parameters to interact_Gxe?
def interact_GxG(pheno,snps1,snps2=None,K=None,covs=None):
    """
    Epistasis test between two sets of SNPs

    Args:
        pheno:  [N x 1] SP.array of 1 phenotype for N individuals
        snps1:  [N x S1] SP.array of S1 SNPs for N individuals
        snps2:  [N x S2] SP.array of S2 SNPs for N individuals
        K:      [N x N] SP.array of LMM-covariance/kinship koefficients (optional)
                        If not provided, then linear regression analysis is performed
        covs:   [N x D] SP.array of D covariates for N individuals

    Returns:
        pv:     [S2 x S1] SP.array of P values for epistasis tests beten all SNPs in
                snps1 and snps2
    """
    if K is None:
        K=SP.eye(N)
    N=snps1.shape[0]
    if snps2 is None:
        snps2 = snps1
    return interact_GxE(snps=snps1,pheno=pheno,env=snps2,covs=covs,K=K)


def interact_GxE_1dof(snps,pheno,env,K=None,covs=None, test='lrt'):
    """
    Univariate GxE fixed effects interaction linear mixed model test for all
    pairs of SNPs and environmental variables.

    Args:
        snps:   [N x S] SP.array of S SNPs for N individuals
        pheno:  [N x 1] SP.array of 1 phenotype for N individuals
        env:    [N x E] SP.array of E environmental variables for N individuals
        K:      [N x N] SP.array of LMM-covariance/kinship koefficients (optional)
                        If not provided, then linear regression analysis is performed
        covs:   [N x D] SP.array of D covariates for N individuals
        test:    'lrt' for likelihood ratio test (default) or 'f' for F-test

    Returns:
        pv:     [E x S] SP.array of P values for interaction tests between all
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
    print(("starting %i interaction scans for %i SNPs each." % (env.shape[1], snps.shape[1])))
    t0=time.time()
    for i in range(env.shape[1]):
        t0_i = time.time()
        cov_i = SP.concatenate((covs,env[:,i:(i+1)]),1)
        lm_i = simple_interaction(snps=snps,pheno=pheno,covs=cov_i,Inter=env[:,i:(i+1)],Inter0=Inter0, test=test)
        pv[i,:]=lm_i.getPv()[0,:]
        t1_i = time.time()
        print(("Finished %i out of %i interaction scans in %.2f seconds."%((i+1),env.shape[1],(t1_i-t0_i))))
    t1 = time.time()
    print(("-----------------------------------------------------------\nFinished all %i interaction scans in %.2f seconds."%(env.shape[1],(t1-t0))))
    return pv


def phenSpecificEffects(snps,pheno1,pheno2,K=None,covs=None,test='lrt'):
    """
    Univariate fixed effects interaction test for phenotype specific SNP effects

    Args:
        snps:   [N x S] SP.array of S SNPs for N individuals (test SNPs)
        pheno1: [N x 1] SP.array of 1 phenotype for N individuals
        pheno2: [N x 1] SP.array of 1 phenotype for N individuals
        K:      [N x N] SP.array of LMM-covariance/kinship koefficients (optional)
                        If not provided, then linear regression analysis is performed
        covs:   [N x D] SP.array of D covariates for N individuals
        test:    'lrt' for likelihood ratio test (default) or 'f' for F-test

    Returns:
        limix LMM object
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
    lm = simple_interaction(snps=Xinter,pheno=Yinter,covs=Covinter,Inter=Inter,Inter0=Inter0,test=test)
    return lm


def simple_interaction(snps,pheno,Inter,Inter0=None,covs = None,K=None,test='lrt'):
    """
    I-variate fixed effects interaction test for phenotype specific SNP effects

    Args:
        snps:   [N x S] SP.array of S SNPs for N individuals (test SNPs)
        pheno:  [N x 1] SP.array of 1 phenotype for N individuals
        Inter:  [N x I] SP.array of I interaction variables to be tested for N
                        individuals (optional). If not provided, only the SNP is
                        included in the null model.
        Inter0: [N x I0] SP.array of I0 interaction variables to be included in the
                         background model when testing for interaction with Inter
        covs:   [N x D] SP.array of D covariates for N individuals
        K:      [N x N] SP.array of LMM-covariance/kinship koefficients (optional)
                        If not provided, then linear regression analysis is performed
        test:    'lrt' for likelihood ratio test (default) or 'f' for F-test

    Returns:
        limix LMM object
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
    if test=='lrt':
        lmi.setTestStatistics(0)
    elif test=='f':
        lmi.setTestStatistics(1)
    else:
        print(test)
        raise NotImplementedError("only f or lrt are implemented")
    lmi.process()
    return lmi


#TOOD: use **kw_args to forward params.. see below
def forward_lmm_kronecker(snps,phenos,Asnps=None,Acond=None,K1r=None,K1c=None,K2r=None,K2c=None,covs=None,Acovs=None,threshold = 5e-8, maxiter = 2,qvalues=False, update_covariances = False,**kw_args):
    """
    Kronecker fixed effects test with forward selection

    Args:
        snps:   [N x S] SP.array of S SNPs for N individuals (test SNPs)
        pheno:  [N x P] SP.array of 1 phenotype for N individuals
        K:      [N x N] SP.array of LMM-covariance/kinship koefficients (optional)
                        If not provided, then linear regression analysis is performed
        covs:   [N x D] SP.array of D covariates for N individuals
        threshold:      (float) P-value thrashold for inclusion in forward selection (default 5e-8)
        maxiter:        (int) maximum number of interaction scans. First scan is
                        without inclusion, so maxiter-1 inclusions can be performed. (default 2)
        qvalues:        Use q-value threshold and return q-values in addition (default False)
        update_covar:   Boolean indicator if covariances should be re-estimated after each forward step (default False)

    Returns:
        lm:             lmix LMMi object
        resultStruct with elements:
            iadded:         array of indices of SNPs included in order of inclusion
            pvadded:        array of Pvalues obtained by the included SNPs in iteration
                            before inclusion
            pvall:   [maxiter x S] SP.array of Pvalues for all iterations
        Optional:      corresponding q-values
            qvadded
            qvall
    """

    #0. checks
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

    covs,Acovs = updateKronCovs(covs,Acovs,N,P)

    if Asnps is None:
        Asnps = [SP.ones([1,P])]
    if (type(Asnps)!=list):
        Asnps = [Asnps]
    assert len(Asnps)>0, "need at least one Snp design matrix"

    if Acond is None:
        Acond = Asnps
    if (type(Acond)!=list):
        Acond = [Acond]
    assert len(Acond)>0, "need at least one Snp design matrix"

    #1. run GP model to infer suitable covariance structure
    if K1c==None or K2c==None:
        vc = estimateKronCovariances(phenos=phenos, K1r=K1r, K2r=K2r, K1c=K1c, K2c=K2c, covs=covs, Acovs=Acovs, **kw_args)
        K1c = vc.getEstTraitCovar(0)
        K2c = vc.getEstTraitCovar(1)
    else:
        vc = None
        assert K1c.shape[0]==P, 'K1c: dimensions dismatch'
        assert K1c.shape[1]==P, 'K1c: dimensions dismatch'
        assert K2c.shape[0]==P, 'K2c: dimensions dismatch'
        assert K2c.shape[1]==P, 'K2c: dimensions dismatch'
    t0 = time.time()
    lm,pv = kronecker_lmm(snps=snps,phenos=phenos,Asnps=Asnps,K1r=K1r,K2r=K2r,K1c=K1c,K2c=K2c,covs=covs,Acovs=Acovs)

    #get pv
    #start stuff
    iadded = []
    pvadded = []
    qvadded = []
    time_el = []
    pvall = SP.zeros((pv.shape[0]*maxiter,pv.shape[1]))
    qvall = None
    t1=time.time()
    print(("finished GWAS testing in %.2f seconds" %(t1-t0)))
    time_el.append(t1-t0)
    pvall[0:pv.shape[0],:]=pv
    imin= SP.unravel_index(pv.argmin(),pv.shape)
    score=pv[imin].min()
    niter = 1
    if qvalues:
        assert pv.shape[0]==1, "This is untested with the fdr package. pv.shape[0]==1 failed"
        qvall = SP.zeros((maxiter,snps.shape[1]))
        qv  = FDR.qvalues(pv)
        qvall[0:1,:] = qv
        score=qv[imin]
    #loop:
    while (score<threshold) and niter<maxiter:
        t0=time.time()
        pvadded.append(pv[imin])
        iadded.append(imin)
        if qvalues:
            qvadded.append(qv[imin])
        if update_covariances and vc is not None:
            vc.addFixedTerm(snps[:,imin[1]:(imin[1]+1)],Acond[imin[0]])
            vc.setScales()#CL: don't know what this does, but findLocalOptima crashes becahuse vc.noisPos=None
            vc.findLocalOptima(fast=True)
            K1c = vc.getEstTraitCovar(0)
            K2c = vc.getEstTraitCovar(1)
            lm.setK1c(K1c)
            lm.setK2c(K2c)
        lm.addCovariates(snps[:,imin[1]:(imin[1]+1)],Acond[imin[0]])
        for i in range(len(Asnps)):
            #add SNP design
            lm.setSNPcoldesign(Asnps[i])
            lm.process()
            pv[i,:] = lm.getPv()[0]
        pvall[niter*pv.shape[0]:(niter+1)*pv.shape[0]]=pv
        imin= SP.unravel_index(pv.argmin(),pv.shape)
        if qvalues:
            qv = FDR.qvalues(pv)
            qvall[niter:niter+1,:] = qv
            score = qv[imin].min()
        else:
            score = pv[imin].min()
        t1=time.time()
        print(("finished GWAS testing in %.2f seconds" %(t1-t0)))
        time_el.append(t1-t0)
        niter=niter+1
    RV = {}
    RV['iadded']  = iadded
    RV['pvadded'] = pvadded
    RV['pvall']   = pvall
    RV['time_el'] = time_el
    if qvalues:
        RV['qvall'] = qvall
        RV['qvadded'] = qvadded
    return lm,RV


def forward_lmm(snps,pheno,K=None,covs=None,qvalues=False,threshold = 5e-8, maxiter = 2,test='lrt',**kw_args):
    """
    univariate fixed effects test with forward selection

    Args:
        snps:   [N x S] SP.array of S SNPs for N individuals (test SNPs)
        pheno:  [N x 1] SP.array of 1 phenotype for N individuals
        K:      [N x N] SP.array of LMM-covariance/kinship koefficients (optional)
                        If not provided, then linear regression analysis is performed
        covs:   [N x D] SP.array of D covariates for N individuals
        threshold:      (float) P-value thrashold for inclusion in forward selection (default 5e-8)
        maxiter:        (int) maximum number of interaction scans. First scan is
                        without inclusion, so maxiter-1 inclusions can be performed. (default 2)
        test:           'lrt' for likelihood ratio test (default) or 'f' for F-test

    Returns:
        lm:             limix LMM object
        iadded:         array of indices of SNPs included in order of inclusion
        pvadded:        array of Pvalues obtained by the included SNPs in iteration
                        before inclusion
        pvall:   [maxiter x S] SP.array of Pvalues for all iterations
    """

    if K is None:
        K=SP.eye(snps.shape[0])
    if covs is None:
        covs = SP.ones((snps.shape[0],1))

    lm = simple_lmm(snps,pheno,K=K,covs=covs,test=test,**kw_args)
    pvall = SP.zeros((maxiter,snps.shape[1]))
    pv = lm.getPv()
    pvall[0:1,:]=pv
    imin= pv.argmin()
    niter = 1
    #start stuff
    iadded = []
    pvadded = []
    qvadded = []
    if qvalues:
        assert pv.shape[0]==1, "This is untested with the fdr package. pv.shape[0]==1 failed"
        qvall = SP.zeros((maxiter,snps.shape[1]))
        qv  = FDR.qvalues(pv)
        qvall[0:1,:] = qv
        score=qv.min()
    else:
        score=pv.min()
    while (score<threshold) and niter<maxiter:
        t0=time.time()
        iadded.append(imin)
        pvadded.append(pv[0,imin])
        if qvalues:
            qvadded.append(qv[0,imin])
        covs=SP.concatenate((covs,snps[:,imin:(imin+1)]),1)
        lm.setCovs(covs)
        lm.process()
        pv = lm.getPv()
        pvall[niter:niter+1,:]=pv
        imin= pv.argmin()
        if qvalues:
            qv = FDR.qvalues(pv)
            qvall[niter:niter+1,:] = qv
            score = qv.min()
        else:
            score = pv.min()
        t1=time.time()
        print(("finished GWAS testing in %.2f seconds" %(t1-t0)))
        niter=niter+1
    RV = {}
    RV['iadded']  = iadded
    RV['pvadded'] = pvadded
    RV['pvall']   = pvall
    if qvalues:
        RV['qvall'] = qvall
        RV['qvadded'] = qvadded
    return lm,RV
