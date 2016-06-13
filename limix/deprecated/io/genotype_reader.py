# Copyright(c) 2014, The LIMIX developers (Christoph Lippert, Paolo Francesco Casale, Oliver Stegle)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import scipy as sp
import limix.deprecated.io.data_util as du
import pandas as pd

class genotype_reader_h5py():
    """
    a genotype reader using the HDF5 interface.
    """

    def __init__(self,file_name):
        self.file_name = file_name
        self.load()

    def load(self,cache_genotype=False,cache_phenotype=True):
        """load data file

        Args:
            cache_genotype:     load genotypes fully into memory (default: False)
            cache_phenotype:    load phentopyes fully intro memry (default: True)
        """
        import h5py
        self.f = h5py.File(self.file_name,'r')
        self.geno  = self.f['genotype']

        #parse out these we always need for convenience
        self.geno_matrix = self.geno['matrix']
        #dimensions
        self.num_samples = self.geno_matrix.shape[0]
        self.num_snps = self.geno_matrix.shape[1]

        self.sample_ID = self.geno['row_header']['sample_ID'][:]
        position   = {
            "chrom":    self.geno['col_header']['chrom'][:],
            "pos":      self.geno['col_header']['pos'][:],
            }
        if 'pos_cum' in list(self.geno['col_header'].keys()):
            position['pos_cum']   = self.geno['col_header']['pos_cum'][:]

        if 'geno_ID' in list(self.geno['col_header'].keys()):
            self.geno_ID   = self.geno['col_header']['geno_ID'][:]
        else:
            self.geno_ID =  sp.arange(self.num_snps)

        #cache?
        if cache_genotype:
            self.geno_matrix = self.geno_matrix[:]
        self.position=pd.DataFrame(data=position,index=self.geno_ID)


    def getGenotypes(self,sample_idx=None,idx_start=None,idx_end=None,chrom=None,pos_start=None,pos_end=None,center=True,unit=True,impute_missing=False,snp_idx=None,windowsize=0):
        """load genotypes.
        Optionally the indices for loading subgroups the genotypes for all people
        based on position of cumulative position.
        Positions can be given as (pos_start-pos_end on chrom)
        If both of these are None (default), then all genotypes are returned

        Args:
            idx_start:         genotype index based selection (start index)
            idx_end:         genotype index based selection (end index)
            chrom:          chromosome based selection (return whole chromosome)
            pos_start:       position based selection (start position) tuple of chrom, position
            pos_end:       position based selection (end position) tuple of chrom, position
            impute_missing: Boolean indicator variable if missing values should be imputed

        Returns:
            X:          scipy.array of genotype values
        """
        #position based matching?
        if (idx_start is None) and (idx_end is None) and ((pos_start is not None) & (pos_end is not None)):
            idx_start,idx_end=self.getGenoIndex(chrom=chrom,pos_start=pos_start,pos_end=pos_end,windowsize=windowsize)
        #index based matching?
        if (idx_start is not None) & (idx_end is not None):
            X = self.geno_matrix[:,idx_start:idx_end]
        elif snp_idx is not None:
            X = self.geno_matrix[:,snp_idx]
        else:
            X = self.geno_matrix[:,:]
        if sample_idx is not None:
            X=X[sample_idx]
        if impute_missing:
            X = du.imputeMissing(X,center=center,unit=unit)
        return X

    def getGenoIndex(self,pos_start=None,pos_end=None,windowsize=0):
        """computes 0-based genotype index from position of cumulative position.
        Positions can be given as (pos_start-pos_end on chrom)
        If both of these are None (default), then all genotypes are returned

        Args:
            chrom:          chromosome based selection (return whole chromosome)
            pos_start:      position based selection (start position) tuple of chrom, position
            pos_end:        position based selection (end position) tuple of chrom, position
            windowsize:     additionally include a flanking window around the selected positions (default 0)

        Returns:
            idx_start:         genotype index based selection (start index)
            idx_end:         genotype index based selection (end index)
        """
        if (pos_start is not None) & (pos_end is not None):
            assert pos_start[0]==pos_end[0],   "getGenoIndex only supports selection on a single chromosome"
            I = self.position["chrom"]==pos_start[0]
            I = I & (self.postion["pos"]>=(pos_start[1]-windowsize)) & (self.position["pos"]<(pos_end[1]+windowsize))
            I =  sp.nonzero(I)[0]
            idx_start = I.min()
            idx_end = I.max()
        elif (chrom is not None):
            I = self.position["chrom"]==chrom

            idx_start = I.min()
            idx_end = I.max()
        else:
            idx_start=None
            idx_end=None
        return idx_start,idx_end

    def getCovariance(self,sample_idx=None,normalize=True,idx_start=None,idx_end=None,pos_start=None,pos_end=None,chrom=None,center=True,unit=True,blocksize=None,X=None,snp_idx=None,windowsize=0,**kw_args):
        """calculate the empirical genotype covariance in a region"""
        if X is not None:
            if X.dtype!= sp.float64:
                X= sp.array(X,dtype= sp.float64)
            K=X.dot(X.T)
            Nsnp=X.shape[1]
        else:
            if (idx_start is None) and (idx_end is None) and ((pos_start is not None) & (pos_end is not None) ) and (chrom is not None):
                idx_start,idx_end=self.getGenoIndex(pos_start=pos_start,pos_end=pos_end,chrom=chrom,pos_cum_start=pos_cum_start,windowsize=windowsize)

            [N,M]=self.geno_matrix.shape
            if blocksize is None:
                blocksize=M
            if idx_start is None:
                idx_start=0
            if idx_end is None:
                idx_end=M
            nread = idx_start
            K=None
            Nsnp=idx_end-idx_start
            while nread<idx_end:
                thisblock=min(blocksize,idx_end-nread)
                X=self.getGenotypes(sample_idx=sample_idx,idx_start=nread,idx_end=(nread+thisblock),center=center,unit=unit,impute_missing=True,**kw_args)
                if X.dtype!= sp.float64:
                    X= sp.array(X,dtype= sp.float64)
                if K is None:
                    K=X.dot(X.T)
                else:
                    K+=X.dot(X.T)
                nread+=thisblock
        if normalize:
            K/=(K.diagonal().mean())
        #else:#divide by number of SNPs in K
        #    K/=Nsnp
        return K

    def getGenoID(self,snp_idx=None,idx_start=None,idx_end=None,pos_start=None,pos_end=None,chrom=None,pos_cum_start=None,pos_cum_end=None):
        """get genotype IDs.
        Optionally the indices for loading subgroups the genotype IDs for all people
        can be given in one out of three ways:
        - 0-based indexing (idx_start-idx_end)
        - position (pos_start-pos_end on chrom)
        - cumulative position (pos_cum_start-pos_cum_end)
        If all these are None (default), then all genotypes are returned

        Args:
            idx_start:         genotype index based selection (start index)
            idx_end:         genotype index based selection (end index)
            pos_start:       position based selection (start position)
            pos_end:       position based selection (end position)
            chrom:      position based selection (chromosome)
            pos_cum_start:   cumulative position based selection (start position)
            pos_cum_end:   cumulative position based selection (end position)

        Returns:
            ID:         scipy.array of genotype IDs (e.g. rs IDs)
        """
        #position based matching?
        if (idx_start is None) and (idx_end is None) and ((pos_start is not None) & (pos_end is not None) & (chrom is not None)) or ((pos_cum_start is not None) & (pos_cum_end is not None)):
            idx_start,idx_end=self.getGenoIndex(pos_start=pos_start,pos_end=pos_end,chrom=chrom,pos_cum_start=pos_cum_start,pos_cum_end=pose_cum1)
        if "genotype_id" in list(self.geno.keys()):
            if (idx_start is not None) & (idx_end is not None):
                return self.geno["genotype_id"][idx_start:idx_end]
            elif snp_idx is not None:
                return self.geno["genotype_id"][snp_idx]
            else:
                return self.geno["genotype_id"][:]
        else:
            if (idx_start is not None) & (idx_end is not None):
                return  sp.arange(idx_start,idx_start)
            elif snp_idx is not None:
                return  sp.arange(self.geno_matrix.shape[1])[snp_idx]
            else:
                return  sp.arange(self.geno_matrix.shape[1])

    def getPos(self,idx_start=None,idx_end=None,pos_start=None,pos_end=None,chrom=None,pos_cum_start=None,pos_cum_end=None):
        """
        get the positions of the genotypes

        Returns:
            chromosome
            position
            cumulative_position
        """
        if (idx_start is None) and (idx_end is None) and ((pos_start is not None) & (pos_end is not None) & (chrom is not None)) or ((pos_cum_start is not None) & (pos_cum_end is not None)):
            idx_start,idx_end=self.getGenoIndex(pos_start=pos_start,pos_end=pos_end,chrom=chrom,pos_cum_start=pos_cum_start,pos_cum_end=pose_cum1)
        if (idx_start is not None) & (idx_end is not None):
            return self.position.iloc[idx_start:idx_end]
        elif snp_idx is not None:
            return self.position.iloc[snp_idx]
        else:
            return self.position


    def getIcis_geno(self,geneID,cis_window=50E3):
        """ if eqtl==True it returns a bool vec for cis """
        assert self.eqtl == True, 'Only for eqtl data'
        index = self.geneID==geneID
        [_chrom,_gene_start,_gene_start] = self.gene_pos[index][0,:]
        Icis = (self.geno_chrom==_chrom)*(self.geno_pos>=_gene_start-cis_window)*(self.geno_pos<=_gene_start+cis_window)
        return Icis

class genotype_reader_tables():
    def __init__(self,file_name):
        self.file_name = file_name
        self.load()

    def load(self,cache_genotype=False,cache_phenotype=True):
        """load data file

        Args:
            cache_genotype:     load genotypes fully into memory (default: False)
            cache_phenotype:    load phentopyes fully intro memry (default: True)
        """
        import tables
        #self.f = h5py.File(self.file_name,'r')
        #self.store = pd.HDFStore(self.file_name,'r')
        self.f = tables.openFile(self.file_name,'r')
        self.geno  = self.f.root.genotype

        #parse out these we always need for convenience
        self.geno_matrix = self.geno.matrix
        #dimensions
        self.num_samples,self.num_snps = self.geno_matrix.shape

        self.sample_ID = self.geno.row_header.sample_ID[:]
        position = {
            "chrom" : self.geno.col_header.chrom[:],
            "pos"   : self.geno.col_header.pos[:],
            }

        if 'pos_cum' in self.geno.col_header:
            position["pos_cum"]   = self.geno.col_header.pos_cum[:]
        #else:
        #    position["geno_pos_cum"] = None

        if 'geno_ID' in self.geno.col_header:
            self.geno_ID   = self.geno.col_header.geno_ID[:]
        else:
            self.geno_ID =  sp.arange(self.num_snps)
        self.position = pd.DataFrame(data=position,index=self.geno_ID)
        #cache?
        if cache_genotype:
            self.geno_matrix = self.geno_matrix[:]



    def getGenotypes(self,sample_idx=None,idx_start=None,idx_end=None,pos_start=None,pos_end=None,chrom=None,center=True,unit=True,pos_cum_start=None,pos_cum_end=None,impute_missing=False,snp_idx=None):
        """load genotypes.
        Optionally the indices for loading subgroups the genotypes for all people
        can be given in one out of three ways:
        - 0-based indexing (idx_start-idx_end)
        - position (pos_start-pos_end on chrom)
        - cumulative position (pos_cum_start-pos_cum_end)
        If all these are None (default), then all genotypes are returned

        Args:
            idx_start:         genotype index based selection (start index)
            idx_end:         genotype index based selection (end index)
            pos_start:       position based selection (start position)
            pos_end:       position based selection (end position)
            chrom:      position based selection (chromosome)
            pos_cum_start:   cumulative position based selection (start position)
            pos_cum_end:   cumulative position based selection (end position)
            impute_missing: Boolean indicator variable if missing values should be imputed

        Returns:
            X:          scipy.array of genotype values
        """
        #position based matching?
        if (idx_start is None) and (idx_end is None) and ((pos_start is not None) & (pos_end is not None) & (chrom is not None)) or ((pos_cum_start is not None) & (pos_cum_end is not None)):
            idx_start,idx_end=self.getGenoIndex(pos_start=pos_start,pos_end=pos_end,chrom=chrom,pos_cum_start=pos_cum_start,pos_cum_end=pose_cum1)
        #index based matching?
        if (idx_start is not None) & (idx_end is not None):
            X = self.geno_matrix[:,idx_start:idx_end]
        elif snp_idx is not None:
            X = self.geno_matrix[:,snp_idx]
        else:
            X = self.geno_matrix[:,:]
        if sample_idx is not None:
            X=X[sample_idx]
        if impute_missing:
            X = du.imputeMissing(X,center=center,unit=unit)
        return X

    def getGenoIndex(self,pos_start=None,pos_end=None,chrom=None,windowsize=0):
        """computes 0-based genotype index from position of cumulative position.
        Positions can be given in one out of two ways:
        - position (pos_start-pos_end on chrom)
        - cumulative position (pos_cum_start-pos_cum_end)
        If all these are None (default), then all genotypes are returned

        Args:
            pos_start:       position based selection (start position)
            pos_end:       position based selection (end position)
            chrom:      position based selection (chromosome)

        Returns:
            idx_start:         genotype index based selection (start index)
            idx_end:         genotype index based selection (end index)
        """
        if (pos_start is not None) & (pos_end is not None):
            assert pos_start[0] == pos_end[0], "chromosomes don't match between start and end position"
            I = self.position["chrom"]==pos_start[0]
            I = I & (self.position["pos"]>=(pos_start[1]-windowsize)) & (self.position["pos"]<(pos_end[1]+windowsize))
            I =  sp.nonzero(I)[0]
            idx_start = I.min()
            idx_end = I.max()
        elif chrom is not None:
            I = self.position["chrom"]==chrom
            I =  sp.nonzero(I)[0]
            if I.size==0:
                return None
            idx_start = I.min()
            idx_end = I.max()
        else:
            idx_start=None
            idx_end=None
        return idx_start,idx_end

    def getCovariance(self,sample_idx=None,normalize=False,idx_start=None,idx_end=None,pos_start=None,pos_end=None,chrom=None,windowsize=0,center=True,unit=True,blocksize=5000,X=None,snp_idx=None,**kw_args):
        """calculate the empirical genotype covariance in a region"""
        if X is not None:
            if X.dtype!= sp.float64:
                X= sp.array(X,dtype= sp.float64)
            K=X.dot(X.T)
            Nsnp=X.shape[1]
        else:
            if (idx_start is None) and (idx_end is None) and ((pos_start is not None) & (pos_end is not None) ) or (chrom is not None):
                idx_start,idx_end=self.getGenoIndex(pos_start=pos_start,pos_end=pos_end,chrom=chrom,windowsize=windowsize)

            [N,M]=self.geno_matrix.shape
            if blocksize is None:
                blocksize=M
            if idx_start is None:
                idx_start=0
            if idx_end is None:
                idx_end=M
            nread = idx_start
            K=None
            Nsnp=idx_end-idx_start
            while nread<idx_end:
                thisblock=min(blocksize,idx_end-nread)
                X=self.getGenotypes(sample_idx=sample_idx,idx_start=nread,idx_end=(nread+thisblock),center=center,unit=unit,impute_missing=True,**kw_args)
                if X.dtype!= sp.float64:
                    X= sp.array(X,dtype= sp.float64)
                if K is None:
                    K=X.dot(X.T)
                else:
                    K+=X.dot(X.T)
                nread+=thisblock
        if normalize:
            K/=(K.diagonal().mean())
        #else:#divide by number of SNPs in K
        #    K/=Nsnp
        return K

    def getGenoID(self,snp_idx=None,idx_start=None,idx_end=None,pos_start=None,pos_end=None,chrom=None,windowsize=0):
        """get genotype IDs.
        Optionally the indices for loading subgroups the genotype IDs for all people
        can be given in one out of three ways:
        - 0-based indexing (idx_start-idx_end)
        - position (pos_start-pos_end on chrom)
        - cumulative position (pos_cum_start-pos_cum_end)
        If all these are None (default), then all genotypes are returned

        Args:
            idx_start:         genotype index based selection (start index)
            idx_end:         genotype index based selection (end index)
            pos_start:       position based selection (start position)
            pos_end:       position based selection (end position)
            chrom:      position based selection (chromosome)

        Returns:
            ID:         scipy.array of genotype IDs (e.g. rs IDs)
        """
        #position based matching?
        if (idx_start is None) and (idx_end is None) and ((pos_start is not None) & (pos_end is not None) ) or (chrom is not None):
            idx_start,idx_end=self.getGenoIndex(pos_start=pos_start,pos_end=pos_end,chrom=chrom,windowsize=windowsize)
        if "genotype_id" in self.geno:
            if (idx_start is not None) & (idx_end is not None):
                return self.geno.genotype_id[idx_start:idx_end]
            elif snp_idx is not None:
                return self.geno.genotype_id[snp_idx]
            else:
                return self.geno.genotype_id[:]
        else:
            if (idx_start is not None) & (idx_end is not None):
                return  sp.arange(idx_start,idx_start)
            elif snp_idx is not None:
                return  sp.arange(self.geno_matrix.shape[1])[snp_idx]
            else:
                return  sp.arange(self.geno_matrix.shape[1])

    def getPos(self,idx_start=None,idx_end=None,snp_idx=None,pos_start=None,pos_end=None,chrom=None,windowsize=0):
        """
        get the positions of the genotypes

        Returns:
            chromosome
            position
            cumulative_position
        """
        if (idx_start is None) and (idx_end is None) and ((pos_start is not None) & (pos_end is not None) ) or (chrom is not None):
            idx_start,idx_end=self.getGenoIndex(pos_start=pos_start,pos_end=pos_end,chrom=chrom,windowsize=windowsize)
        if (idx_start is not None) & (idx_end is not None):
            return self.position.iloc[idx_start:idx_end]
        elif snp_idx is not None:
            return self.position.iloc[snp_idx]
        else:
            return self.position


"""
    def to_table(self, filename):
        file = tables.open
        pass

class Genotype(tables.IsDescription):
    index       = tables.UInt64Col()    # running index (64-bit integer)
    id          = tables.StringCol(16)  # e.g. rs-id  (16-character String)
    chromosome  = tables.UInt8Col()     # Unsigned short integer
    bp_position = tables.UInt32Col()    # unsigned byte
    values      = tables.CArray()

"""
