"""@package util
misc. utility functions used in limix modules and demos
"""

import numpy as np
import scipy as sp
import scipy as SP
import pdb, sys, pickle
import matplotlib.pylab as plt
import scipy.stats as st
import scipy.interpolate

# def qvalues1(PV,m=None,pi=1.0):
#     """estimate q vlaues from a list of Pvalues
#     this algorihm is taken from Storey, significance testing for genomic ...
#     m: number of tests, (if not len(PV)), pi: fraction of expected true null (1.0 is a conservative estimate)
#     """
          
#     S = PV.shape
#     PV = PV.flatten()
#     if m is None:
#         m = len(PV) * 1.0
#     else:
#         m*=1.0
#     lPV = len(PV)
    
#     #1. sort pvalues
#     PV = PV.squeeze()
#     IPV = PV.argsort()
#     PV  = PV[IPV]

#     #2. estimate lambda
#     if pi is None:
#         lrange = sp.linspace(0.05,0.95,max(lPV/100.0,10))
#         pil    = sp.double((PV[:,sp.newaxis]>lrange).sum(axis=0))/lPV
#         pilr   = pil/(1.0-lrange)
#         #ok, I think for SNPs this is pretty useless, pi is close to 1!
#         pi =1.0
#         #if there is something useful in there use the something close to 1
#         if pilr[-1]<1.0:
#             pi = pilr[-1]
            
#     #3. initialise q values
#     QV_ = pi * m/lPV* PV
#     QV_[-1] = min(QV_[-1],1.0)
#     #4. update estimate
#     for i in xrange(lPV-2,-1,-1):
#         QV_[i] = min(pi*m*PV[i]/(i+1.0),QV_[i+1])
#     #5. invert sorting
#     QV = sp.zeros_like(PV)
#     QV[IPV] = QV_

#     QV = QV.reshape(S)
#     return QV
    
# def qvalues(pv, m = None, return_pi0 = False, lowmem = False, pi0 = None, fix_lambda = None):

#     original_shape = pv.shape

#     assert(pv.min() >= 0 and pv.max() <= 1)

#     pv = pv.ravel() # flattens the array in place, more efficient than flatten() 

#     if m == None:
# 	m = float(len(pv))
#     else:
# 	# the user has supplied an m, let's use it
# 	m *= 1.0

#     # if the number of hypotheses is small, just set pi0 to 1
#     if len(pv) < 100:
# 	pi0 = 1.0
#     elif pi0 != None:
# 	pi0 = pi0
#     else:
# 	# evaluate pi0 for different lambdas
# 	pi0 = []
# 	lam = sp.arange(0, 0.90, 0.01)
# 	counts = sp.array([(pv > i).sum() for i in sp.arange(0, 0.9, 0.01)])
	
# 	if fix_lambda != None:
# 	    interv_count = (pv > fix_lambda - 0.01).sum()
# 	    uniform_sim = sp.array([(pv > fix_lambda-0.01).sum()*(i+1) for i in sp.arange(0, len(sp.arange(0, 0.90, 0.01)))][::-1])
# 	    counts += uniform_sim
	    
# 	for l in range(len(lam)):
# 	    pi0.append(counts[l]/(m*(1-lam[l])))

# 	pi0 = sp.array(pi0)

# 	# fit natural cubic spline
# 	tck = sp.interpolate.splrep(lam, pi0, k = 3)
# 	pi0 = sp.interpolate.splev(lam[-1], tck)
# 	if pi0 > 1:
# 	    print ("got pi0 > 1 (%.3f) while estimating qvalues, setting it to 1" % pi0)
# 	    pi0 = 1.0

# 	assert(pi0 >= 0 and pi0 <= 1), "%f" % pi0

#     if lowmem:
# 	# low memory version, only uses 1 pv and 1 qv matrices
# 	qv = sp.zeros((len(pv),))
# 	last_pv = pv.argmax()
# 	qv[last_pv] = (pi0*pv[last_pv]*m)/float(m)
# 	pv[last_pv] = -sp.inf
# 	prev_qv = last_pv
# 	for i in xrange(int(len(pv))-2, -1, -1):
# 	    cur_max = pv.argmax()
# 	    qv_i = (pi0*m*pv[cur_max]/float(i+1))
# 	    pv[cur_max] = -sp.inf
# 	    qv_i1 = prev_qv
# 	    qv[cur_max] = min(qv_i, qv_i1)
# 	    prev_qv = qv[cur_max]

#     else:
# 	p_ordered = sp.argsort(pv)    
# 	pv = pv[p_ordered]
# 	# estimate qvalues
# # 	qv = pi0*m*pv/(sp.arange(len(pv))+1.0)
	
# # 	for i in xrange(int(len(qv))-2, 0, -1):
# # 	    qv[i] = min([qv[i], qv[i+1]])


# 	qv = pi0 * m/len(pv) * pv
# 	qv[-1] = min(qv[-1],1.0)

# 	for i in xrange(len(pv)-2, -1, -1):
# 	    qv[i] = min(pi0*m*pv[i]/(i+1.0), qv[i+1])

# 	# reorder qvalues
# 	qv_temp = qv.copy()
# 	qv = sp.zeros_like(qv)
# 	qv[p_ordered] = qv_temp

#     # reshape qvalues
#     qv = qv.reshape(original_shape)

#     if return_pi0:
# 	return qv, pi0
#     else:
# 	return qv

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

def manhattanplot(pval,chromosome,position,pv_max=0.1,qv=None,alpha=0.05):
    """plot manhattanplot
    qv: q-values, only used to determine significane threshold
    alpha: significance threshold. If no q-values provided we use Bonferroni
    """

    #1. determine significance threshold
    if qv is None:
        qv = pval*pval.shape[0]

    #2. get max pv that is still significance
    pv_max_sig = pval[qv<alpha]

    if len(pv_max_sig)>0:
        pv_max_sig = pv_max_sig.max()
    else:
        pv_max_sig = None

    Iplot = (pval<pv_max)
    pval = pval[Iplot]
    chromosome = chromosome[Iplot]
    position = position[Iplot]
    
    chromosomes = sp.unique(chromosome)
    posmax  = sp.zeros(chromosomes.shape)
    posmax_cumsum = 0
    position_plot = position.copy()

    # loop over chromosomes
    for i in xrange(chromosomes.shape[0]):
        i_chr = chromosome == chromosomes[i]
        posmax[i] = position[i_chr].max()
        position_plot[i_chr] += posmax_cumsum
        posmax_cumsum+=posmax[i]
        if sp.mod(i,2)>0:
            plt.plot(position_plot[i_chr],-sp.log10(pval[i_chr]),'.k')
        else:
            plt.plot(position_plot[i_chr],-sp.log10(pval[i_chr]),'.b')

    #axes labels
    plt.xlim([position_plot.min(),position_plot.max()])
    _ylim = plt.ylim()
    plt.ylim([-SP.log10(pv_max),_ylim[1]])
    plt.xlabel('Genomic position')
    plt.ylabel('-Log10 PV')
    #significance threshold
    if pv_max_sig:
        plt.hlines(-SP.log10(pv_max_sig),plt.xlim()[0],plt.xlim()[1])


    
    

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

def getPosNew(data):
    """
    get Fixed position
    """
    pos = data.geno['col_header']['pos'][:]
    chrom= data.geno['col_header']['chrom'][:]
    n_chroms = chrom.max()
    pos_new = []
    for chrom_i in range(1,n_chroms+1):
        I = chrom==chrom_i
        _pos = pos[I]
        for i in range(1,_pos.shape[0]):
            if not _pos[i]>_pos[i-1]:
                _pos[i:]=_pos[i:]+_pos[i-1]
        pos_new.append(_pos)
    pos_new = SP.concatenate(pos_new)
    return pos_new

def getCumPos(data):
    """
    getCumulativePosition
    """
    pos = getPosNew(data)
    chrom= data.geno['col_header']['chrom'][:]
    n_chroms = int(chrom.max())
    x = 0
    for chrom_i in range(1,n_chroms+1):
        I = chrom==chrom_i
        pos[I]+=x
        x=pos[I].max()
    return pos

def getChromBounds(data):
    """
    getChromBounds
    """
    chrom= data.geno['col_header']['chrom'][:]
    posCum = getCumPos(data)
    n_chroms = int(chrom.max())
    chrom_bounds = []
    for chrom_i in range(2,n_chroms+1):
        I1 = chrom==chrom_i
        I0 = chrom==chrom_i-1
        _cb = 0.5*(posCum[I0].max()+posCum[I1].min())
        chrom_bounds.append(_cb)
    return chrom_bounds