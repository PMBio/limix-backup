import numpy as np
import scipy as sp
import pdb, sys, pickle
import matplotlib.pylab as plt
import scipy.stats

def get_column(filename, col, skiprows = 1, dtype = float):
    import pdb
    f = open(filename, 'r')
    results = []
    skipped = 0  
    for line in f:
        if skipped < skiprows:
            skipped += 1
            continue
        data = sp.array(line.strip('\n').split('\t'))
        results.append(data[col])
        
    try:
        results = np.asarray(results, dtype = dtype)
    except ValueError:
        results = np.asarray(results, dtype = str)
        results = np.asarray(results[results != 'NA'], dtype = dtype)

    return results

def estimate_lambda(pv):
    LOD2 = sp.median(sp.stats.chi2.isf(pv, 1))
    L = (LOD2/0.456)
    return L

def qqplot_bar(M=1000000, alphaLevel = 0.05, distr = 'log10'):
    #assumes 'log10'   
    import scipy as SP
    import pylab as PL
    import scipy.stats as ST
    #pdb.set_trace()
    mRange=10**(SP.arange(SP.log10(0.5),SP.log10(M-0.5)+0.1,0.1));#should be exp or 10**?
    numPts=len(mRange);
    betaalphaLevel=SP.zeros(numPts);#down in the plot
    betaOneMinusalphaLevel=SP.zeros(numPts);#up in the plot
    betaInvHalf=SP.zeros(numPts);
    for n in xrange(numPts):
        m=mRange[n]; #numPLessThanThresh=m;
        betaInvHalf[n]=ST.beta.ppf(0.5,m,M-m);
        betaalphaLevel[n]=ST.beta.ppf(alphaLevel,m,M-m);
        betaOneMinusalphaLevel[n]=ST.beta.ppf(1-alphaLevel,m,M-m);
        pass
    betaDown=betaInvHalf-betaalphaLevel;
    betaUp=betaOneMinusalphaLevel-betaInvHalf;

    theoreticalPvals=mRange/M;
    return betaUp, betaDown, theoreticalPvals
    #pdb.set_trace()
    #PL.figure()
    #PL.plot(-SP.log10(theoreticalPvals),-SP.log10(theoreticalPvals + betaUp));
    #PL.plot(-SP.log10(theoreticalPvals),-SP.log10(theoreticalPvals - betaDown));
    #PL.show()
    

def qqplot(pval, filename = None, distr = 'log10', alphaLevel = 0.05):
    tests = pval.shape[0]
    pnull = (0.5 + sp.arange(tests))/tests
    # pnull = np.sort(np.random.uniform(size = tests))    
    
    if distr == 'chi2':    
        qnull = sp.stats.chi2.isf(pnull, 1)   
        qemp = (sp.stats.chi2.isf(sp.sort(pval),1))
        xl = 'LOD scores'
        yl = '$\chi^2$ quantiles'
    
    if distr == 'log10':
        qnull = -sp.log10(pnull)
        qemp = -sp.log10(sp.sort(pval))
        
        xl = '-log10(P) observed'
        yl = '-log10(P) expected'

    plt.figure()
    plt.plot(qnull, qemp, '.')
    #plt.plot([0,qemp.max()], [0,qemp.max()],'r')
    plt.plot([0,qnull.max()], [0,qnull.max()],'r')
    plt.ylabel(xl)
    plt.xlabel(yl)
    plt.title('Genomic control: %.3f' % estimate_lambda(pval.flatten()))
    if alphaLevel is not None:
        if distr == 'log10':
            betaUp, betaDown, theoreticalPvals = qqplot_bar(M=tests,alphaLevel=alphaLevel,distr=distr)
            lower = -sp.log10(theoreticalPvals-betaDown)
            upper = -sp.log10(theoreticalPvals+betaUp)
            plt.plot(-sp.log10(theoreticalPvals),lower,'g-.')
            plt.plot(-sp.log10(theoreticalPvals),upper,'g-.')
       
    if filename != None:
        plt.savefig(filename) 

def manhattanplot(pval,chromosome,position):
    plt.figure()
    chromosomes = sp.unique(chromosome)
    for i in xrange(chromosomes.shape[0]):
        i_chr = chromosome == chromosomes[i]
        posmax = position[i_chr.max]
    #add nice axis labels

# def mean_impute(X, imissX=None, maxval=2.0):
#     if imissX is None:
#         imissX = np.isnan(X)
    
#     n_i,n_s=X.shape
#     if imissX is None:
#         n_obs_SNP=np.ones(X.shape)
#     else:    
#         i_nonan=(~imissX)
#         n_obs_SNP=i_nonan.sum(0)
#         X[imissX]=0.0
#     snp_sum=(X).sum(0)
#     one_over_sqrt_pi=(1.0+snp_sum)/(2.0+maxval*n_obs_SNP)
#     one_over_sqrt_pi=1./np.sqrt(one_over_sqrt_pi*(1.-one_over_sqrt_pi))
#     snp_mean=(snp_sum*1.0)/(n_obs_SNP)

#     # X_ret=X-snp_mean
#     for i in range(X.shape[1]):
#         nan = np.isnan(X[:, i])
#         i_nonan = (~nan)
#         X[i_nonan,i] -= X[i_nonan, i].mean()
#         X[i_nonan,i] /= X[i_nonan, i].std()

#     #X[i_nonan] = X[i_nonan].mean(axis=0)
#     X_ret = X/np.sqrt(X.shape[1])
#     # X_ret*=one_over_sqrt_pi
#     if imissX is not None:
#         X_ret[imissX]=0.0
#     return X_ret


def mean_impute(X, imissX=None, maxval=2.0):
    if imissX is None:
        imissX = np.isnan(X)
    
    n_i,n_s=X.shape
    if imissX is None:
        n_obs_SNP=np.ones(X.shape)
    else:    
        i_nonan=(~imissX)
        n_obs_SNP=i_nonan.sum(0)
        X[imissX]=0.0
    snp_sum=(X).sum(0)
    one_over_sqrt_pi=(1.0+snp_sum)/(2.0+maxval*n_obs_SNP)
    one_over_sqrt_pi=1./np.sqrt(one_over_sqrt_pi*(1.-one_over_sqrt_pi))
    snp_mean=(snp_sum*1.0)/(n_obs_SNP)

    X_ret=X-snp_mean
    X_ret*=one_over_sqrt_pi
    if imissX is not None:
        X_ret[imissX]=0.0
    return X_ret
