# Copyright(c) 2014, The LIMIX developers (Christoph Lippert, Paolo Francesco Casale, Oliver Stegle)
# All rights reserved.
#
# LIMIX is provided under a 2-clause BSD license.
# See license.txt for the complete license.

import scipy as SP
import copy

import limix.modules.data_util as du
import limix.modules.genotype_reader as gr
import limix.modules.phenotype_reader as pr
import pandas as pd

class QTLData(object):
    """
    This class is a mask to a the contents of a geno_reader and a pheno_reader or a subset thereof
    """

    def __init__(self,geno_reader=None,pheno_reader=None):
        self.geno_reader = geno_reader
        self.pheno_reader = pheno_reader
        #self.geno_pos["chrom"] = self.geno_reader.geno_chrom
        self.geno_pos = self.geno_reader.position
        self.geno_ID = self.geno_reader.geno_ID
        self.phenotype_ID = self.pheno_reader.phenotype_ID
        self.geno_snp_idx = None      #SNP indices
        self.sample_idx = du.merge_indices([self.geno_reader.sample_ID, self.pheno_reader.sample_ID],header=["geno","pheno"],join="inner")      #index of individuals
        self.sample_ID = self.geno_reader.sample_ID[SP.array(self.sample_idx["geno"])]
        self.num_snps = self.geno_reader.num_snps
        

    def range_query_geno_local(self, idx_start=None, idx_end=None, chrom=None, pos_start=None, pos_end=None, pos_cum_start=None, pos_cum_end=None):
        """
        return an index for a range query on the genotypes
        """
        if idx_start==None and idx_end==None and pos_start==None and pos_end==None and chrom==None and pos_cum_start==None and pos_cum_end==None:
            return SP.arange(0,self.num_snps)
        elif idx_start is not None or idx_end is not None:
            if idx_start is None:
                idx_start = 0
            if idx_end is None:
                idx_end = self.num_snps
            res = SP.arange(idx_start,idx_end)
            return res
        elif chrom is not None:
            idx_chr = self.geno_pos["chrom"]==chrom
            if pos_start is None:
                idx_larger = SP.ones(self.num_snps,dtype=bool)
            else:
                idx_larger = self.geno_pos["pos"]>=pos_start
            if pos_end is None:
                idx_smaller = SP.ones(self.num_snps,dtype=bool)
            else:
                idx_smaller = self.geno_pos["pos"]<=pos_end
            res = idx_chr & idx_smaller & idx_larger

        elif pos_cum_start is not None or pos_cum_end is not None:
            if pos_cum_start is None:
                idx_larger = SP.ones(self.num_snps,dtype=bool)
            else:
                idx_larger = self.geno_pos["pos_cum"]>=pos_cum_start
            if pos_cum_end is None:
                idx_smaller = SP.ones(self.num_snps,dtype=bool)
            else:
                idx_smaller = self.geno_pos["pos_cum"]<=pos_cum_end
            res = idx_smaller & idx_larger
        else:
            raise Exception("This should not be triggered")#res = SP.ones(self.geno_pos.shape,dtype=bool)
        return SP.where(res)[0]
        

    def range_query_geno(self, idx_start=None, idx_end=None, chrom=None, pos_start=None, pos_end=None, pos_cum_start=None, pos_cum_end=None):
        """
        return an index for a range query on the genotypes
        """
        if idx_start==None and idx_end==None and pos_start==None and pos_end==None and chrom==None and pos_cum_start==None and pos_cum_end==None:
            return self.geno_snp_idx
        else:
            res = self.range_query_geno_local(idx_start=idx_start, idx_end=idx_end, chrom=chrom, pos_start=pos_start, pos_end=pos_end, pos_cum_start=pos_cum_start, pos_cum_end=pos_cum_end)
        if self.geno_snp_idx is None:
            return SP.where(res)[0]
        else:    
            return self.geno_snp_idx[res]

            
    def getGenotypes(self,idx_start=None,idx_end=None,pos_start=None,pos_end=None,chrom=None,center=True,unit=True,pos_cum_start=None,pos_cum_end=None,impute_missing=False):
        """return genotypes. 
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
        query_idx = self.range_query_geno(idx_start=idx_start, idx_end=idx_end, chrom=chrom, pos_start=pos_start, pos_end=pos_end, pos_cum_start=pos_cum_start, pos_cum_end=pos_cum_end)
        X = self.geno_reader.getGenotypes(sample_idx=SP.array(self.sample_idx["geno"]),snp_idx=query_idx) 
        if impute_missing:
            X = du.imputeMissing(X,center=center,unit=unit)
        return X

    def getCovariance(self,normalize=False,idx_start=None,idx_end=None,pos_start=None,pos_end=None,chrom=None,center=True,unit=True,pos_cum_start=None,pos_cum_end=None,blocksize=None,X=None,**kw_args):
        """calculate the empirical genotype covariance in a region"""
        return self.geno_reader.getCovariance(sample_idx=SP.array(self.sample_idx["geno"]),normalize=normalize,idx_start=idx_start,idx_end=idx_end,pos_start=pos_start,pos_end=pos_end,chrom=chrom,center=center,unit=unit,pos_cum_start=pos_cum_start,pos_cum_end=pos_cum_end,blocksize=blocksize,X=X,**kw_args)

    def getGenoID(self,idx_start=None,idx_end=None,pos_start=None,pos_end=None,chrom=None,pos_cum_start=None,pos_cum_end=None):
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
        query_idx = self.range_query_geno_local(idx_start=idx_start, idx_end=idx_end, chrom=chrom, pos_start=pos_start, pos_end=pos_end, pos_cum_start=pos_cum_start, pos_cum_end=pos_cum_end)
        if query_idx is None:
            return self.geno_ID
        else:
            return self.geno_ID[query_idx]

       

    def getPhenotypes(self,phenotype_IDs=None,phenotype_query=None,center=True,intersection=False):
        """load Phenotypes
        
        Args:
            idx_start:      phenotype indices to load (start individual index)
            idx_end:       phenotype indices to load (end individual index)
            phenotype_IDs:  names of phenotypes to load
            impute:         imputation of missing values (default: True)
            intersection:   restrict observation to those obseved in all phenotypes (true) or at least in one phenotype (false)? (default: False)
        
        Returns:
            phenotypes:     phenotype values
            sample_idx_intersect:        index of individuals in phenotypes after filtering missing values
        """

        phenotypes, sample_idx_intersect = self.pheno_reader.getPhenotypes(sample_idx=SP.array(self.sample_idx["pheno"]),phenotype_IDs=phenotype_IDs,phenotype_query=phenotype_query,center=center,intersection=intersection)
        return phenotypes, sample_idx_intersect

    def getPos(self,idx_start=None,idx_end=None,pos_start=None,pos_end=None,chrom=None,pos_cum_start=None,pos_cum_end=None):
        """
        get the positions of the genotypes

        Returns:
            chromosome
            position
            cumulative_position
        """
        query_idx = self.range_query_geno_local(idx_start=idx_start, idx_end=idx_end, chrom=chrom, pos_start=pos_start, pos_end=pos_end, pos_cum_start=pos_cum_start, pos_cum_end=pos_cum_end)
        if query_idx is None:            
            return self.geno_pos
        else:
            return self.geno_pos.iloc[query_idx]

    def subsample(self,rows=None,cols_pheno=None,cols_geno=None,idx_start=None,idx_end=None,pos_start=None,pos_end=None,chrom=None,pos_cum_start=None,pos_cum_end=None):
        """sample a particular set of individuals (rows) or phenotypes (cols_pheno) or genotypes (cols_geno)
        
        Args:
            rows:           indices for a set of individuals
            cols_pheno:     indices for a set of phenotypes
            cols_geno:      indices for a set of SNPs
        
        Returns:
            QTLdata object holding the specified subset of the data
        """
        if not (idx_start==None and idx_end==None and pos_start==None and pos_end==None and chrom==None and pos_cum_start==None and pos_cum_end==None):
            query_idx = self.range_query_geno_local(idx_start=idx_start, idx_end=idx_end, chrom=chrom, pos_start=pos_start, pos_end=pos_end, pos_cum_start=pos_cum_start, pos_cum_end=pos_cum_end)
            return self.subsample(rows=rows,cols_pheno=cols_pheno,cols_geno=res,idx_start=None,idx_end=None,pos_start=None,pos_end=None,chrom=None,pos_cum_start=None,pos_cum_end=None)
        C = copy.copy(self)
        
        if rows is not None:
            C.sample_ID = C.sample_ID[rows]         #IDs of individuals
            C.sample_idx = C.sample_idx.iloc[rows]   #index of individuals
        
        if cols_geno is not None:
            assert cols_geno.dtype=="int"
            C.geno_pos = C.geno_pos.iloc[cols_geno]
            C.geno_ID = C.geno_ID[cols_geno]
            if C.geno_snp_idx is not None:
                C.geno_snp_idx = C.geno_snp_idx[cols_geno]
            else:
                 C.geno_snp_idx = cols_geno
            C.num_snps=len(C.geno_snp_idx)

        if cols_pheno is not None:
            C.phenotype_ID = C.phenotype_ID[cols_pheno]

        return C

    def subsample_phenotypes(self,phenotype_IDs=None,phenotype_query=None,center=True,intersection=False):
        """load Phenotypes
        
        Args:
            idx_start:      phenotype indices to load (start individual index)
            idx_end:       phenotype indices to load (end individual index)
            phenotype_IDs:  names of phenotypes to load
            center:         imputation of missing values (default: True)
            intersection:   restrict observation to those obseved in all phenotypes (true) or at least in one phenotype (false)? (default: False)
        
        Returns:
            phenotypes:     phenotype values
            sample_idx_intersect:        index of individuals in phenotypes after filtering missing values
        """

        phenotypes, sample_idx_intersect = self.pheno_reader.getPhenotypes(phenotype_query=phenotype_query,sample_idx=SP.array(self.sample_idx["pheno"]),phenotype_IDs=phenotype_IDs,center=center,intersection=intersection)
        return self.subsample(rows=sample_idx_intersect,cols_pheno=None,cols_geno=None,idx_start=None,idx_end=None,pos_start=None,pos_end=None,chrom=None,pos_cum_start=None,pos_cum_end=None)
            