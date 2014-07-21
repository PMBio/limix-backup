import sys
import scipy as SP
import pdb
import pylab as PL
import copy
import os
import cPickle
import glob

def plot_manhattan(plt,posCum,pv,chromBounds,
					thr=None,qv=None,lim=None,xticklabels=True,
					alphaNS=0.1,alphaS=0.5,colorNS='DarkBlue',colorS='Orange'):
	"""
	This script makes a manhattan plot
	-------------------------------------------
	plt				matplotlib.axes.AxesSubplot
	posCum			cumulative position
	pv				pvalues
	chromBounds		chrom boundaries
	qv				qvalues
					if provided, threshold for significance is set on qvalues but pvalues are plotted
	thr				threshold for significance
					default: 0.01 bonferroni correceted significance levels if qvs are not specified,
					or 0.01 on qvs if qvs specified
	lim				top limit on y-axis
					if not provided, -1.2*log(pv.min()) is taken
	xticklabels		if true, xtick labels are printed
	alphaNS			transparency of non-significant SNPs
	alphaS			transparency of significant SNPs
	"""

	if thr==None:
		thr = 0.01/float(posCum.shape[0])

	if lim==None:
		lim=-1.2*SP.log10(SP.minimum(pv.min(),thr))

	chromBounds = SP.concatenate([chromBounds,SP.array([posCum.max()])])

	n_chroms = chromBounds.shape[0]
	for chrom_i in range(0,n_chroms-1,2):
		PL.fill_between(posCum,0,lim,where=(posCum>chromBounds[chrom_i]) & (posCum<chromBounds[chrom_i+1]),facecolor='LightGray',linewidth=0,alpha=0.5)

	if qv==None:
		Isign = pv<thr
	else:
		Isign = pv<thr

	PL.plot(posCum[~Isign],-SP.log10(pv[~Isign]),'.',color=colorNS,ms=5,alpha=alphaNS)
	PL.plot(posCum[Isign], -SP.log10(pv[Isign]), '.',color=colorS,ms=5,alpha=alphaS)

	PL.plot([0,posCum.max()],[-SP.log10(thr),-SP.log10(thr)],'--',color='Gray')

	PL.ylim(0,lim)

	PL.ylabel('-log$_{10}$pv')
	PL.xlim(0,posCum.max())
	xticks = SP.array([chromBounds[i:i+2].mean() for i in range(chromBounds.shape[0]-1)])
	plt.set_xticks(xticks)
	PL.xticks(fontsize=6)

	if xticklabels:
		plt.set_xticklabels(SP.arange(1,n_chroms+1))
		PL.xlabel('genetic position')
	else:
		plt.set_xticklabels([])

	plt.spines["right"].set_visible(False)
	plt.spines["top"].set_visible(False)
	plt.xaxis.set_ticks_position('bottom')
	plt.yaxis.set_ticks_position('left')


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
