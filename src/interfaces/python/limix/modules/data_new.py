import scipy as SP
import h5py
import copy

import data_util as du

import genotype_reader as gr
import phenotype_reader as pr

class QTLData(object):
    """
    This class is a mask to a the contents of a geno_reader and a pheno_reader or a subset thereof
    """

    def __init__(self,geno_reader=None,pheno_reader=None):
        self.geno_reader = geno_reader
        self.pheno_reader = pheno_reader
        self.geno_chrom = None
        self.geno_pos = None
        self.geno_id = None
        self.phenotype_id = None
        self.geno_snp_idx = None      #SNP indices
        self.geno_snp_idx_reverse = None  #SNP indices in original array
        self.geno_ind_idx = None      #index of individuals
        self.pheno_ind_idx = None     #index of individuals
        self.sample_ID = None

    def range_query_pos(self, chrom, pos_start=None, pos_stop=None):
        """
        return an index for a range query on the genotypes
        """
        idx_chr = self.genoChrom==chrom
        if pos_start is None and pos_stop is None:
            self.pass

        else:
            pass
 
    def getGenotypes(self,i0=None,i1=None,pos0=None,pos1=None,chrom=None,center=True,unit=True,pos_cum0=None,pos_cum1=None,impute_missing=True):
        """return genotypes. 
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

        query_idx = self.intersect_index(self.geno_snp_idx)
        if self.geno_snp_idx is None:
            #index based matching?
            X = self.geno_reader.read(i0=i0,i1=i1,pos0=pos0,pos1=pos1,chrom=chrom,pos_cum0=pos_cum0,pos_cum1=pos_cum1)
        elif i0==None and i1==None and pos0==None and pos1==None and chrom==None and pos_cum0==None and pos_cum1==None:
            X = self.geno_reader.read(snp_idx=query_idx)
        else:
            #TODO: intersect mask with index
            raise NotImplementedError("not implemented yet")
        if impute_missing:
            X = du.imputeMissing(X,center=center,unit=unit)
        return X

    def getCovariance(self,normalize=True,i0=None,i1=None,pos0=None,pos1=None,chrom=None,center=True,unit=True,pos_cum0=None,pos_cum1=None,blocksize=None,X=None,**kw_args):
        """calculate the empirical genotype covariance in a region"""
        raise NotImplementedError("not implemented yet")

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
        raise NotImplementedError("TODO")

    def getPhenotypes(self,phenotype_IDs=None,center=True,impute=True,intersection=False):
        """load Phenotypes
        
        Args:
            i0:             phenotype indices to load (start individual index)
            i1:             phenotype indices to load (stop individual index)
            phenotype_IDs:  names of phenotypes to load
            impute:         imputation of missing values (default: True)
            intersection:   restrict observation to those obseved in all phenotypes? (default: False)
        
        Returns:
            Y:              phenotype values
            Ikeep:          index of individuals in Y
        """
        raise NotImplementedError("TODO")

    def getPos(self):
        """
        get the positions of the genotypes

        Returns:
            chromosome
            position
            cumulative_position
        """
        return [self.genoChrom,self.genoPos,self.genoPos_cum]

    def subSample(self,Irow=None,Icol_geno=None,Icol_pheno=None):
        """sample a particular set of individuals (Irow) or phenotypes (Icol_pheno) or genotypes (Icol_geno)
        
        Args:
            Irow:           indices for a set of individuals
            Icol_pheno:     indices for a set of phenotypes
            Icol_geno:      indices for a set of SNPs
        
        Returns:
            QTLdata opject holding the specified subset of the data
        """
        C = copy.copy(self)
        if Irow is not None:
            C.sample_ID = C.sample_ID[Irow]   #IDs of individuals
            C.geno_ind_idx = C.geno_ind_idx   #index of individuals
            C.pheno_ind_idx = C.pheno_ind_idx #index of individuals
            
        if Icol_geno is not None:
            C.genoPos = C.genoPos[Icol_geno]
            C.genoChrom = C.genoChrom[Icol_geno]
            C.genoID=C.genoID[Icol_geno]
            C.genoPos_cum = C.genoPos_cum[Icol_geno]
            if C.geno_snp_idx is not None:
                C.geno_snp_idx[~C.self.geno_snp_idx_reverse[IcolGeno]]=False
            else:
                 C.geno_snp_idx=SP.zeros(C.genoPos.shape[1])
            C.self.geno_snp_idx_reverse = self.geno_snp_idx_reverse[IcolGeno]
        if Icol_pheno is not None:
            C.phenotype_ID = C.phenotype_ID[Icol_pheno]
        return C
