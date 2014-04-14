import scipy as SP
import h5py
import copy
try:
    #see if fastlmm is in the path for the fast C-based parser
    import fastlmm.pyplink.wrap_plink_parser as parser  
    cparser=True
except:
    cparser=False
    pass

def estCumPos(pos,chrom,offset = 20000000):
    '''
    compute the cumulative position of each variant given the position and the chromosome
    Also return the starting cumulativeposition of each chromosome
    '''
    chromvals = SP.unique(chrom)#SP.unique is always sorted
    chrom_pos=SP.zeros_like(chromvals)#get the starting position of each Chrom
    cum_pos = SP.zeros_like(pos)#get the cum_pos of each variant.
    maxpos_cum=0
    for i,mychrom in enumerate(chromvals):
        chrom_pos[i] = maxpos_cum
        i_chr=chrom==mychrom
        maxpos = pos[i_chr].max()+offset
        maxpos_cum+=maxpos
        cum_pos[i_chr]=chrom_pos[i]+pos[i_chr]
    return cum_pos,chrom_pos
        
        
def imputeMissing(X,center=True,unit=True):
        '''
        fill in missing values in the SNP matrix by the mean value
        optionally center the data and unit-variance it
        '''
        typeX=X.dtype
        if typeX!=SP.int8:
            iNanX = X!=X
        else:
            iNanX = X==-9
        if iNanX.any():
            
            if cparser:
                betaNotUnitVariance=False
                betaA=1.0
                betaB=1.0
                print "using C-based imputer"
                if X.flags["C_CONTIGUOUS"] or typeX!=SP.float32:
                    X = SP.array(X, order="F", dtype=SP.float32)
                    if typeX==SP.int8:
                        X[iNanX]=SP.nan
                    parser.standardize(X,betaNotUnitVariance=betaNotUnitVariance,betaA=betaA,betaB=betaB)
                    X=SP.array(X,dtype=SP.float64)
                else:
                    parser.standardize(X,betaNotUnitVariance=betaNotUnitVariance,betaA=betaA,betaB=betaB)
                X=SP.array(X,dtype=SP.float64)
            else:
                nObsX = (~iNanX).sum(0)
                if typeX!=SP.float64:
                    X=SP.array(X,dtype=SP.float64)
                X[iNanX] = 0.0
                sumX = (X).sum(0)                
                meanX = sumX/nObsX
                if center:
                    X-=meanX
                    X[iNanX] = 0.0
                    X_=X
                else:
                    mean=SP.tile(meanX,(X.shape[0],1))
                    X[iNanX]=mean[iNanX]
                    X_=X-mean
                if unit:
                    stdX = SP.sqrt((X_*X_).sum(0)/nObsX)
                    stdX[stdX==0.0]=1.0
                    X/=stdX
        else:
            if X.dtype!=SP.float64:
                X=SP.array(X,dtype=SP.float64)
            if center:
                X-= X.mean(axis=0)
            if unit:
                stdX= X.std(axis=0)
                stdX[stdX==0.0]=1.0
                X/=stdX
        return X

class QTLData():
    def __init__(self,file_name):
        self.file_name = file_name
        self.load()

    def load(self,cache_genotype=False,cache_phenotype=True):
        """load data file:
        cache_genotype: load genotypes fully into memory (False)
        cache_phenotype: load phentopyes fully intro memry (True)        
        """
        self.f = h5py.File(self.file_name,'r')
        self.pheno = self.f['phenotype']
        self.geno  = self.f['genotype']
        #TODO: load all row and column headers for genotype and phenotype
        
        #parse out thse we alwasy need for convenience
        self.genoM = self.geno['matrix']
        self.phenoM = self.pheno['matrix']
        self.sample_ID = self.geno['row_header']['sample_ID'][:]
        self.genoChrom = self.geno['col_header']['chrom'][:]
        self.genoPos   = self.geno['col_header']['pos'][:]
        if 'pos_cum' in self.geno['col_header'].keys():
            self.genoPos_cum   = self.geno['col_header']['pos_cum'][:]
        else:
            self.genoPos_cum = None
        self.phenotype_ID = self.pheno['col_header']['phenotype_ID'][:]


        #cache?
        if cache_genotype:
            self.genoM = self.genoM[:]
        if cache_phenotype:
            self.phenoM = self.phenoM[:]
    
        # Additional pheno col header
        headers = self.pheno['col_header'].keys()
        if 'gene_ID' in headers:
            self.eqtl = True
            self.geneID = self.pheno['col_header']['gene_ID'][:]
            self.gene_pos = SP.array([self.pheno['col_header']['gene_chrom'][:],self.pheno['col_header']['gene_start'][:],self.pheno['col_header']['gene_end']],dtype='int').T
            self.geneIDs= list(set(self.geneID))
        else:
            self.eqtl = False
        if 'environment' in headers:
            self.E  = self.pheno['col_header/environment'][:]
            self.Es = list(set(self.E))
        else:
            self.E = None

        #dimensions
        self.N = self.genoM.shape[0]
        self.S = self.genoM.shape[1]
        self.P = self.phenoM.shape[1]
        assert (self.genoM.shape[0]==self.phenoM.shape[0]), 'dimension missmatch'

    def getGenoIndex(self,pos0=None,pos1=None,chrom=None,pos_cum0=None,pos_cum1=None):
        if (pos0 is not None) & (pos1 is not None) & (chrom is not None):
            I = self.gneoChrom==chrom
            I = I & (self.genoPos>=p0) & (self.genoPos<p1)
            I = SP.nonzero(I)[0]
            i0 = I.min()
            i1 = I.max()
        elif (pos_cum0 is not None) & (pos_cum1 is not None):
            I = (self.genoPos_cum>=pos_cum0) & (self.genoPos_cum<pos_cum1)
            I = SP.nonzero(I)[0]
            if I.size==0:
                return None
            i0 = I.min()
            i1 = I.max()
        else:
            i0=None
            i1=None
        return i0,i1
 
    def getGenotypes(self,i0=None,i1=None,pos0=None,pos1=None,chrom=None,center=True,unit=True,pos_cum0=None,pos_cum1=None):
        """load genotypes
        i0..i1: genotype index based selection
        pos0..pos1 / chrom: position based selection
        pos_cum0..pos_cum1: cumulative position based selection
        """
        #position based matching?
        if (i0 is None) and (i1 is None) and ((pos0 is not None) & (pos1 is not None) & (chrom is not None)) or ((pos_cum0 is not None) & (pos_cum1 is not None)):
            i0,i1=self.getGenoIndex(pos0=pos0,pos1=pos1,chrom=chrom,pos_cum0=pos_cum0,pos_cum1=pose_cum1)
        #index based matching?
        if (i0 is not None) & (i1 is not None):
            X = self.genoM[:,i0:i1]
        else:
            X = self.genoM[:,:]
        X = imputeMissing(X,center=center,unit=unit)
        return X

    def getCovariance(self,normalize=True,i0=None,i1=None,pos0=None,pos1=None,chrom=None,center=True,unit=True,pos_cum0=None,pos_cum1=None,blocksize=None,X=None,**kw_args):
        """calculate the empirical genotype covariance in a region"""
        if X is not None:
            K=X.dot(X.T)
            Nsnp=X.shape[1]
        else:
            if (i0 is None) and (i1 is None) and ((pos0 is not None) & (pos1 is not None) & (chrom is not None)) or ((pos_cum0 is not None) & (pos_cum1 is not None)):
                i0,i1=self.getGenoIndex(pos0=pos0,pos1=pos1,chrom=chrom,pos_cum0=pos_cum0,pos_cum1=pose_cum1)

            [N,M]=self.genoM.shape
            if blocksize is None:
                blocksize=M
            if i0 is None:
                i0=0
            if i1 is None:
                i1=M
            nread = i0
            K=None
            Nsnp=i1-i0
            while nread<i1:
                thisblock=min(blocksize,i1-nread)
                X=self.getGenotypes(i0=nread,i1=(nread+thisblock),center=center,unit=unit,**kw_args)    
                if K is None:
                    K=X.dot(X.T)
                else:
                    K+=X.dot(X.T)
                nread+=thisblock
        if normalize:
            K/=(K.diagonal().mean())
        else:#divide by number of SNPs in K
            K/=Nsnp
        return K

    def getPhenotypes(self,i0=None,i1=None,phenotype_IDs=None,center=True,impute=True,intersection=False):
        """load Phenotypes
        i0..i1: phenotype indices to load
        phenotype_IDs: names of phenotypes to load
        impute: imputation of missing values (True)
        intersection: restrict observation to those obseved in all phenotypes? (False)
        """
        if phenotype_IDs is not None:
            I = SP.array([SP.nonzero(self.phenotype_ID==n)[0][0] for n in phenotype_IDs])
        else:
            I = SP.arange(self.phenotype_ID.shape[0])
        Y = SP.array(self.phenoM[:,I],dtype='float')
        Iok = (~SP.isnan(Y))

        if intersection:
            Ikeep = Iok.all(axis=1)
        else:
            Ikeep = Iok.any(axis=1)

        Y = Y[Ikeep]
        Iok = Iok[Ikeep]

        if center | impute:
            for i in xrange(Y.shape[1]):
                ym = Y[Iok[:,i],i].mean()
                Y[:,i]-=ym
                Y[~Iok[:,i],i] = ym
                Y[:,i]/=Y[:,i].std()
        #calculate overlap of missing values
        return [Y,Ikeep]

    def getPos(self):
        return [self.genoChrom,self.genoPos,self.genoPos_cum]

    def getIcis_geno(self,geneID,cis_window=50E3):
        """ if eqtl==True it returns a bool vec for cis """
        assert self.eqtl == True, 'Only for eqtl data'
        index = self.geneID==geneID
        [_chrom,_gene_start,_gene_end] = self.gene_pos[index][0,:]
        Icis = (self.genoChrom==_chrom)*(self.genoPos>=_gene_start-cis_window)*(self.genoPos<=_gene_end+cis_window)
        return Icis

    def subSample(self,Irow=None,Icol_geno=None,Icol_pheno=None):
        """sample a particular set of individuals (Irow) or phenotypes (Icol_pheno) or genotypes (Icol_geno)"""
        C = copy.copy(self)
        if Irow is not None:
            C.genoM = C.genoM[Irow]
            C.phenoM = C.phenoM[Irow]
            C.sample_ID = C.sample_ID[Irow]
        if Icol_geno is not None:
            C.genoM = C.genoM[:,Icol_geno]
            C.genoPos = C.genoPos[Icol_geno]
            C.genoChrom = C.genoChrom[Icol_geno]
            C.genoPos_cum = C.genoPos_cum[Icol_geno]
        if Icol_pheno is not None:
            C.phenoM = C.phenoM[:,Icol_pheno]
            C.phenotype_ID = C.phenotype_ID[Icol_pheno]
            if C.eqtl:
                C.geneID   = C.geneID[Icol_pheno]
                C.gene_pos = C.gene_pos[Icol_pheno,:]
                C.geneIDs  = list(set(C.geneID))
            if C.E != None:
                C.E  = C.E[Icol_pheno]
                C.Es = list(set(C.E))

        C.N = C.genoM.shape[0]
        C.S = C.genoM.shape[1]
        C.P = C.phenoM.shape[1]
        return C

