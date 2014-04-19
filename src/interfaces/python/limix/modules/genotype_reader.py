import scipy as SP
import h5py
import data_util as du

class genotype_reader_hdf5():
    def __init__(self,file_name):
        self.file_name = file_name
        self.load()

    def load(self,cache_genotype=False,cache_phenotype=True):
        """load data file
        
        Args:
            cache_genotype:     load genotypes fully into memory (default: False)
            cache_phenotype:    load phentopyes fully intro memry (default: True)        
        """
        self.f = h5py.File(self.file_name,'r')
        self.geno  = self.f['genotype']
        #TODO: load all row and column headers for genotype and phenotype
        
        #parse out thse we alwasy need for convenience
        self.genoM = self.geno['matrix']
        self.sample_ID = self.geno['row_header']['sample_ID'][:]
        self.genoChrom = self.geno['col_header']['chrom'][:]
        self.genoPos   = self.geno['col_header']['pos'][:]
        if 'pos_cum' in self.geno['col_header'].keys():
            self.genoPos_cum   = self.geno['col_header']['pos_cum'][:]
        else:
            self.genoPos_cum = None
        
        #cache?
        if cache_genotype:
            self.genoM = self.genoM[:]
        
        #dimensions
        self.N = self.genoM.shape[0]
        self.S = self.genoM.shape[1]
        
    def getGenoIndex(self,pos0=None,pos1=None,chrom=None,pos_cum0=None,pos_cum1=None):
        """computes 0-based genotype index from position of cumulative position. 
        Positions can be given in one out of two ways: 
        - position (pos0-pos1 on chrom)
        - cumulative position (pos_cum0-pos_cum1)
        If all these are None (default), then all genotypes are returned

        Args:
            pos0:       position based selection (start position)
            pos1:       position based selection (stop position)
            chrom:      position based selection (chromosome)
            pos_cum0:   cumulative position based selection (start position)
            pos_cum1:   cumulative position based selection (stop position)
        
        Returns:
            i0:         genotype index based selection (start index)
            i1:         genotype index based selection (stop index)
        """
        if (pos0 is not None) & (pos1 is not None) & (chrom is not None):
            I = self.genoChrom==chrom
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
 
    def getGenotypes(self,i0=None,i1=None,pos0=None,pos1=None,chrom=None,center=True,unit=True,pos_cum0=None,pos_cum1=None,impute_missing=True,snp_idx=None):
        """load genotypes. 
        Optionally the indices for loading subgroups the genotypes for all people
        can be given in one out of three ways: 
        - 0-based indexing (i0-i1)
        - position (pos0-pos1 on chrom)
        - cumulative position (pos_cum0-pos_cum1)
        If all these are None (default), then all genotypes are returned

        Args:
            i0:         genotype index based selection (start index)
            i1:         genotype index based selection (stop index)
            pos0:       position based selection (start position)
            pos1:       position based selection (stop position)
            chrom:      position based selection (chromosome)
            pos_cum0:   cumulative position based selection (start position)
            pos_cum1:   cumulative position based selection (stop position)
            impute_missing: Boolean indicator variable if missing values should be imputed
        
        Returns:
            X:          scipy.array of genotype values
        """
        #position based matching?
        if (i0 is None) and (i1 is None) and ((pos0 is not None) & (pos1 is not None) & (chrom is not None)) or ((pos_cum0 is not None) & (pos_cum1 is not None)):
            i0,i1=self.getGenoIndex(pos0=pos0,pos1=pos1,chrom=chrom,pos_cum0=pos_cum0,pos_cum1=pose_cum1)
        #index based matching?
        if (i0 is not None) & (i1 is not None):
            X = self.genoM[:,i0:i1]
        elif snp_idx is not None:
            X = self.genoM[:,snp_idx]
        else:
            X = self.genoM[:,:]
        if impute_missing:
            X = du.imputeMissing(X,center=center,unit=unit)
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

    def getGenoID(self,i0=None,i1=None,pos0=None,pos1=None,chrom=None,pos_cum0=None,pos_cum1=None):
        """get genotype IDs. 
        Optionally the indices for loading subgroups the genotype IDs for all people
        can be given in one out of three ways: 
        - 0-based indexing (i0-i1)
        - position (pos0-pos1 on chrom)
        - cumulative position (pos_cum0-pos_cum1)
        If all these are None (default), then all genotypes are returned

        Args:
            i0:         genotype index based selection (start index)
            i1:         genotype index based selection (stop index)
            pos0:       position based selection (start position)
            pos1:       position based selection (stop position)
            chrom:      position based selection (chromosome)
            pos_cum0:   cumulative position based selection (start position)
            pos_cum1:   cumulative position based selection (stop position)
           
        Returns:
            ID:         scipy.array of genotype IDs (e.g. rs IDs)
        """
        #position based matching?
        if (i0 is None) and (i1 is None) and ((pos0 is not None) & (pos1 is not None) & (chrom is not None)) or ((pos_cum0 is not None) & (pos_cum1 is not None)):
            i0,i1=self.getGenoIndex(pos0=pos0,pos1=pos1,chrom=chrom,pos_cum0=pos_cum0,pos_cum1=pose_cum1)
        if "genotype_id" in self.geno.keys():
            if (i0 is not None) & (i1 is not None):
                return self.geno["genotype_id"][i0:i1]
            else:
                return self.geno["genotype_id"][i0:i1]
        else:
            if (i0 is not None) & (i1 is not None):
                return SP.arange(i0,i0)
            else:
                return SP.arange(self.genoM.shape[1])

    def getPos(self):
        """
        get the positions of the genotypes

        Returns:
            chromosome
            position
            cumulative_position
        """
        return [self.genoChrom,self.genoPos,self.genoPos_cum]

    def getIcis_geno(self,geneID,cis_window=50E3):
        """ if eqtl==True it returns a bool vec for cis """
        assert self.eqtl == True, 'Only for eqtl data'
        index = self.geneID==geneID
        [_chrom,_gene_start,_gene_end] = self.gene_pos[index][0,:]
        Icis = (self.genoChrom==_chrom)*(self.genoPos>=_gene_start-cis_window)*(self.genoPos<=_gene_end+cis_window)
        return Icis