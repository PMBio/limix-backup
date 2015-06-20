import sys
sys.path.append('./..')
import h5py
import pdb
import scipy as SP
import scipy.stats as ST
import scipy.linalg as LA
import time as TIME
import copy
import warnings
import os


class Splitter():

	def __init__(self,pos=None,chrom=None,separate_chroms=False):
		"""
		Constructor
		Args:
			pos:		position
			chrom:		chromosome
		"""
		# assert
		assert pos is not None, 'Slider:: set pos'
		assert chrom is not None, 'Slider:: set chrom'

		self.pos   = pos
		self.chrom = chrom
		# sep chroms
		self.separate_chroms = separate_chroms
		# windows
		self.windows = None
		# additional info for windows
		self.info	= {}
		pass

	def splitGeno(self,method='slidingWindow',size=5e4,step=None,annotation_file=None,cis=1e4,funct=None,minSnps=1.,maxSnps=SP.inf,cache=False,out_dir='./cache',fname=None,rewrite=False):
		"""
		split into windows
		Args:
			method:		method used to slit the windows:
						'slidingWindow':	uses a sliding window
						'geneWindow':		uses windows centered on genes
			size:		window size used in slidingWindow method
			step:		moving step used in slidingWindow method
			annotation_file:	file containing the annotation file for geneWindow method
			minSnps:	only windows with nSnps>=minSnps are considered
			maxSnps:	only windows with nSnps>=maxSnps are considered
			cache:		if results need to be cashed
			out_dir:	folder used for caching
			fname:		name of the hdf5 file used for caching
			rewrite:	if true, rewrite the existing cache hdf5 file
		"""
		self.info= {} # forget everything
		# check if it is necessary to read form file or not
		read_from_file = False
		if cache:
			assert fname is not None, 'Splitter:: specify fname'
			if not os.path.exists(out_dir):
				os.makedirs(out_dir)
			out_file = os.path.join(out_dir,fname)
			read_from_file = os.path.exists(out_file) and not rewrite

		self.size = size

		if read_from_file:
			# reads from file
			f = h5py.File(out_file,'r')
			self.windows = f['windows'][:]
			for key in f.keys():
				if key is 'windows': continue
				self.info[key] = f[key][:] 
			f.close()
		else:
			# calculates windows using the indicated method
			if method=='slidingWindow':
				self.windows = self._splitGenoSlidingWindow(size=size,step=step,minSnps=minSnps,maxSnps=maxSnps)
			elif method=='geneWindow':
				self.windows = self._splitGenoGeneWindow(annotation_file=annotation_file,cis=cis,funct=funct,minSnps=minSnps,maxSnps=maxSnps)
			if cache:
				f = h5py.File(out_file,'w')
				f.create_dataset('windows',data=self.windows)
				for key in self.info.keys():
					f.create_dataset(key,data=self.info[key])
				f.close()

	def _splitGenoSlidingWindow(self,size=5e4,step=None,minSnps=1.,maxSnps=SP.inf):
		"""
		split into windows using a slide criterion
		Args:
			size:		window size
			step:		moving step (default: 0.5*size)
			minSnps:	only windows with nSnps>=minSnps are considered
			maxSnps:	only windows with nSnps>=maxSnps are considered
		"""
		if step is None:	step = 0.5*size
		chroms = SP.unique(self.chrom)
		windows = []
		nSnps   = []

		for chrom_i in chroms:
			print 'spitting chrom %s' % chrom_i
			start = 0
			while 1:
				if start>self.pos[self.chrom==chrom_i].max(): break
				wnd = [chrom_i,start,start+size]
				Ir = (self.chrom==wnd[0])*(self.pos>=wnd[1])*(self.pos<wnd[2])
				_nSnps = Ir.sum()
				if _nSnps>=minSnps and _nSnps<=maxSnps:
					windows.append(wnd)
					nSnps.append(_nSnps)
				start += step
		self.info['nSnps'] = SP.array(nSnps)
		return SP.array(windows)

	def _splitGenoGeneWindow(self,annotation_file=None,cis=1e4,funct='protein_coding',minSnps=1.,maxSnps=SP.inf):
		"""
		split into windows based on genes
		"""
		 #1. load annotation
		assert annotation_file is not None, 'Splitter:: specify annotation file'
		try:
			f = h5py.File(annotation_file,'r')
			geneID = f['geneID'][:]
			gene_chrom = f['chrom'][:]
			gene_start = f['start'][:]
			gene_end = f['end'][:]
			gene_strand = f['strand'][:]
			gene_function = f['function'][:]
			f.close()
		except:
			print 'Splitter:: format annotation file not valid'

		# if funct is not None, it has to be a list
		if funct is not None and funct!=list:	funct=[funct]

		windows = []
		nSnps   = []
		Igene   = []
		#2. calculates windows 
		for gene_i in range(geneID.shape[0]):
			if funct is not None:
				if gene_function[gene_i] not in funct:
					Igene.append(False)
					continue
			wnd = [gene_chrom[gene_i],gene_start[gene_i]-cis,gene_end[gene_i]+cis]
			Ir = (self.chrom==wnd[0])*(self.pos>=wnd[1])*(self.pos<=wnd[2])
			_nSnps = Ir.sum()
			if _nSnps>=minSnps and _nSnps<=maxSnps:
				windows.append(wnd)
				nSnps.append(_nSnps)
				Igene.append(True)
			else:
				Igene.append(False)
		Igene = SP.array(Igene)
		self.info['nSnps'] = SP.array(nSnps)
		self.info['geneID']        = geneID[Igene]
		self.info['gene_start']    = gene_start[Igene]
		self.info['gene_end']      = gene_end[Igene]
		self.info['gene_strand']   = gene_strand[Igene]
		self.info['gene_function'] = gene_function[Igene]
		return SP.array(windows)

	def getWindow(self,i):
		"""
		get window number i
		Returns:
			Ir:		boolean vector that marks SNPs in cis
			info:	other useful info about the region
		"""
		assert self.windows is not None, 'split Genome before!'
		wnd = self.windows[i,:]
		if self.separate_chroms:
			pos = self.pos[self.chrom==wnd[0]]
			Ir = (pos>=wnd[1])*(pos<=wnd[2])
		else:
			Ir = (self.chrom==wnd[0])*(self.pos>=wnd[1])*(self.pos<=wnd[2])

		info = {'chrom':wnd[0:1],'start':wnd[1:2],'end':wnd[2:3],'pos':SP.array([wnd[1:3].mean()])}
		for key in self.info.keys():
			info[key] = SP.array([self.info[key][i]])
		return Ir,info

	def getWindowSingleSnp(self,i):
		"""
		get window number i
		Returns:
			Ir:		boolean vector that marks SNPs in cis
		"""
		assert self.windows is not None, 'split Genome before!'
		warnings.warn('work only if step size is half the window size')
		warnings.warn('DEPRECATED')
		wnd = self.windows[i,:]
		Ichrom = self.chrom==wnd[0]
		pos = self.pos[Ichrom]
		# first window in the chromosome?
		if i==0:							first_chrom = True
		elif wnd[0]!=self.windows[i-1,0]:	first_chrom=True
		else:								first_chrom=False
		# last window in the chromosome?
		if i==self.windows.shape[0]-1:		last_chrom = True
		elif wnd[0]!=self.windows[i+1,0]:	last_chrom = True
		else:								last_chrom = False
		# select stuff
		start = wnd[1]+0.25*self.size
		end   = wnd[2]-0.25*self.size
		if first_chrom:		start = wnd[1]
		if last_chrom:		end   = SP.inf
		if self.separate_chroms:
			Ir = (pos>=start)*(pos<end)
		else:
			Ir = Ichrom
			Ir[Ichrom] = (pos>=start)*(pos<end)

		info = {'chrom':wnd[0:1],'start':SP.array([start]),'end':SP.array([end])}
		return Ir,info

	def get_nWindows(self):
		"""
		get number of windows
		"""
		return float(self.windows.shape[0])

