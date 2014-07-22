import sys
import scipy as sp 
import pdb
import pylab as pl 
import matplotlib.pylab as plt
import scipy.stats as st
import copy
import os
import cPickle
import glob

def plot_manhattan(posCum,pv,chromBounds,
					thr=None,qv=None,lim=None,xticklabels=True,
					alphaNS=0.1,alphaS=0.5,colorNS='DarkBlue',colorS='Orange',plt=None):
	"""
	This script makes a manhattan plot
	-------------------------------------------
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
	plt				matplotlib.axes.AxesSubplot, the target handle for this figure (otherwise current axes)
	"""
	if plt is None:
		plt = pl.gca()

	if thr==None:
		thr = 0.01/float(posCum.shape[0])

	if lim==None:
		lim=-1.2*sp.log10(sp.minimum(pv.min(),thr))

	chromBounds = sp.concatenate([chromBounds,sp.array([posCum.max()])])

	n_chroms = chromBounds.shape[0]
	for chrom_i in range(0,n_chroms-1,2):
		pl.fill_between(posCum,0,lim,where=(posCum>chromBounds[chrom_i]) & (posCum<chromBounds[chrom_i+1]),facecolor='LightGray',linewidth=0,alpha=0.5)

	if qv==None:
		Isign = pv<thr
	else:
		Isign = pv<thr

	pl.plot(posCum[~Isign],-sp.log10(pv[~Isign]),'.',color=colorNS,ms=5,alpha=alphaNS)
	pl.plot(posCum[Isign], -sp.log10(pv[Isign]), '.',color=colorS,ms=5,alpha=alphaS)

	pl.plot([0,posCum.max()],[-sp.log10(thr),-sp.log10(thr)],'--',color='Gray')

	pl.ylim(0,lim)

	pl.ylabel('-log$_{10}$pv')
	pl.xlim(0,posCum.max())
	xticks = sp.array([chromBounds[i:i+2].mean() for i in range(chromBounds.shape[0]-1)])
	plt.set_xticks(xticks)
	pl.xticks(fontsize=6)

	if xticklabels:
		plt.set_xticklabels(sp.arange(1,n_chroms+1))
		pl.xlabel('genetic position')
	else:
		plt.set_xticklabels([])

	plt.spines["right"].set_visible(False)
	plt.spines["top"].set_visible(False)
	plt.xaxis.set_ticks_position('bottom')
	plt.yaxis.set_ticks_position('left')


def _qqplot_bar(M=1000000, alphaLevel = 0.05, distr = 'log10'):
	"""calculate theoretical expectations for qqplot"""
	mRange=10**(sp.arange(sp.log10(0.5),sp.log10(M-0.5)+0.1,0.1));#should be exp or 10**?
	numPts=len(mRange);
	betaalphaLevel=sp.zeros(numPts);#down in the plot
	betaOneMinusalphaLevel=sp.zeros(numPts);#up in the plot
	betaInvHalf=sp.zeros(numPts);
	for n in xrange(numPts):
	   m=mRange[n]; #numPLessThanThresh=m;
	   betaInvHalf[n]=st.beta.ppf(0.5,m,M-m);
	   betaalphaLevel[n]=st.beta.ppf(alphaLevel,m,M-m);
	   betaOneMinusalphaLevel[n]=st.beta.ppf(1-alphaLevel,m,M-m);
	betaDown=betaInvHalf-betaalphaLevel;
	betaUp=betaOneMinusalphaLevel-betaInvHalf;

	theoreticalPvals=mRange/M;
	return betaUp, betaDown, theoreticalPvals
    

def qqplot(pv, distr = 'log10', alphaLevel = 0.05):
	"""
	This script makes a QQ plot
	-------------------------------------------
	pv				pvalues (numpy array)
	distr           scale of the distribution (log10 or chi2)
	alphaLevel      significance bounds
	"""
	shape_ok = (len(pv.shape)==1) or ((len(pv.shape)==2) and pv.shape[1]==1)
	assert shape_ok, 'qqplot requires a 1D array of p-values'

	tests = pv.shape[0]
	pnull = (0.5 + sp.arange(tests))/tests
	# pnull = np.sort(np.random.uniform(size = tests))    
	Ipv = sp.argsort(pv)

	if distr == 'chi2':    
	    qnull = sp.stats.chi2.isf(pnull, 1)   
	    qemp = (sp.stats.chi2.isf(pv[Ipv],1))
	    xl = 'LOD scores'
	    yl = '$\chi^2$ quantiles'

	if distr == 'log10':
	    qnull = -sp.log10(pnull)
	    qemp = -sp.log10(pv[Ipv])
	    
	    xl = '-log10(P) observed'
	    yl = '-log10(P) expected'

	plt.plot(qnull, qemp, '.')
	#plt.plot([0,qemp.m0x()], [0,qemp.max()],'r')
	plt.plot([0,qnull.max()], [0,qnull.max()],'r')
	plt.ylabel(xl)
	plt.xlabel(yl)
	if alphaLevel is not None:
	    if distr == 'log10':
	        betaUp, betaDown, theoreticalPvals = _qqplot_bar(M=tests,alphaLevel=alphaLevel,distr=distr)
	        lower = -sp.log10(theoreticalPvals-betaDown)
	        upper = -sp.log10(theoreticalPvals+betaUp)
	        plt.fill_between(-sp.log10(theoreticalPvals),lower,upper,color='grey',alpha=0.5)
	        #plt.plot(-sp.log10(theoreticalPvals),lower,'g-.')
	        #plt.plot(-sp.log10(theoreticalPvals),upper,'g-.')
