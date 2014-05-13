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
		PL.fill_between(posCum,0,lim,where=(posCum>chromBounds[chrom_i])*(posCum<chromBounds[chrom_i+1]),facecolor='LightGray',linewidth=0,alpha=0.5)

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

