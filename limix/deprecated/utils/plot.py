# Copyright(c) 2014, The LIMIX developers (Christoph Lippert, Paolo Francesco Casale, Oliver Stegle)
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import sys
import scipy as sp
import numpy as np
import pdb
import pylab as pl
import matplotlib.pylab as plt
import scipy.stats as st
import copy
import os
import pickle
import glob

def plot_manhattan(posCum,pv,chromBounds=None,
					thr=None,qv=None,lim=None,xticklabels=True,
					alphaNS=0.1,alphaS=0.5,colorNS='DarkBlue',colorS='Orange',plt=None,thr_plotting=None,labelS=None,labelNS=None):
	"""
	This script makes a manhattan plot
	-------------------------------------------
	posCum			cumulative position
	pv				pvalues
	chromBounds		chrom boundaries (optionally). If not supplied, everything will be plotted into a single chromosome
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
	thr_plotting	plot only P-values that are smaller than thr_plotting to speed up plotting
    labelS           optional plotting label (significant loci)
    labelNS          optional plotting label (non significnat loci)
	"""
	if plt is None:
		plt = pl.gca()

	if thr==None:
		thr = 0.01/float(posCum.shape[0])

	if lim==None:
		lim=-1.2*sp.log10(sp.minimum(pv.min(),thr))

	if chromBounds is None:
		chromBounds = sp.array([[0,posCum.max()]])
	else:
		chromBounds = sp.concatenate([chromBounds,sp.array([posCum.max()])])


	n_chroms = chromBounds.shape[0]
	for chrom_i in range(0,n_chroms-1,2):
		pl.fill_between(posCum,0,lim,where=(posCum>chromBounds[chrom_i]) & (posCum<chromBounds[chrom_i+1]),facecolor='LightGray',linewidth=0,alpha=0.5)

	if thr_plotting is not None:
		if pv is not None:
			i_small = pv<thr_plotting
		elif qv is not None:
			i_small = qv<thr_plotting

		if qv is not None:
			qv = qv[i_small]
		if pv is not None:
			pv = pv[i_small]
		if posCum is not None:
			posCum=posCum[i_small]

	if qv==None:
		Isign = pv<thr
	else:
		Isign = qv<thr

	pl.plot(posCum[~Isign],-sp.log10(pv[~Isign]),'.',color=colorNS,ms=5,alpha=alphaNS,label=labelNS)
	pl.plot(posCum[Isign], -sp.log10(pv[Isign]), '.',color=colorS,ms=5,alpha=alphaS,label=labelS)

	if qv is not None:
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
	for n in range(numPts):
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
	This script makes a Quantile-Quantile plot of the observed
	negative log P-value distribution against the theoretical one under the null.

	Input:
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

	line = plt.plot(qnull, qemp, '.')[0]
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
	return line


def plot_normal(x=None, mean_x=None,std_x=None,color='red',linewidth=2,alpha=1,bins=20,xlim=False,plot_mean=True,plot_std=False,plot_2std=True,figure=None,annotate=True,histogram=True):
    """
    plot a fit of a normal distribution to the data in x.
    """
    import pylab
    if figure is None:
        figure=pylab.figure()
    if mean_x is None:
        #fit maximum likelihood Normal distribution mean to samples X
        mean_x = x.mean() #sample mean
    if std_x is None:
        #fit maximum likelihood Normal distribution standard deviation to samples X
        std_x = x.std()   #sample standard deviation

    xvals=np.arange(mean_x-5*std_x,mean_x+5*std_x,.001)
    yvals=st.norm.pdf(xvals,mean_x,std_x)
    #plot normal distribution:
    ax = pylab.plot(xvals,yvals,color=color,linewidth=linewidth,alpha=alpha)
    if x is not None and histogram:
        #plot histogram of x-values
        pylab.hist(x,bins,normed=True)

    if plot_mean:
        #evaluate distribution at the mean:
        max_cdf=st.norm.pdf(mean_x,mean_x,std_x)
        pylab.plot([mean_x,mean_x],[0,max_cdf],color=color,linewidth=linewidth,alpha=alpha,linestyle="--")
        if annotate:
            pylab.annotate('$\mu$', xy=(mean_x+0.6*std_x, 1.0*max_cdf),
                horizontalalignment='center', verticalalignment='center',fontsize=15,color=color)
    if plot_std:#plot mean +- 1*standard deviation (64% interval)
        std_cdf=st.norm.pdf(mean_x+std_x,mean_x,std_x)
        pylab.plot([mean_x+std_x,mean_x+std_x],[0,std_cdf],color=color,linewidth=linewidth,alpha=alpha,linestyle="--")
        pylab.plot([mean_x-std_x,mean_x-std_x],[0,std_cdf],color=color,linewidth=linewidth,alpha=alpha,linestyle="--")
        if annotate:
            pylab.annotate('$\mu+\sigma$', xy=(mean_x+1.6*std_x, 1.5*std_cdf),
                horizontalalignment='center', verticalalignment='center',fontsize=15,color=color)
    if plot_2std:#plot mean +- 2*standard deviations (95% interval)
        std2_cdf=st.norm.pdf(mean_x+2*std_x,mean_x,std_x)
        pylab.plot([mean_x+2*std_x,mean_x+2*std_x],[0,std2_cdf],color=color,linewidth=linewidth,alpha=alpha,linestyle="--")
        pylab.plot([mean_x-2*std_x,mean_x-2*std_x],[0,std2_cdf],color=color,linewidth=linewidth,alpha=alpha,linestyle="--")
        if annotate:
            pylab.annotate('$\mu+2\sigma$', xy=(mean_x+2.6*std_x, 1.5*std2_cdf),
                horizontalalignment='center', verticalalignment='center',fontsize=15,color=color)
    if xlim: #cut of unused space on y-axis
        pylab.xlim([mean_x-4*std_x,mean_x+4*std_x])
    return figure
