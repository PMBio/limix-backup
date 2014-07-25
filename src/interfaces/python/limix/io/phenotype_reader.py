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

import scipy as SP
import limix.io.data_util as du
import pandas as pd

class pheno_reader_tables():
    def __init__(self,file_name):
        self.file_name = file_name
        self.load()

    def load(self,cache_phenotype=True):
        """load data file
        
        Args:
            cache_phenotype:    load phentopyes fully intro memry (default: True)        
        """
        import tables
        self.f = tables.openFile(self.file_name,'r')
        self.pheno = self.f.root.phenotype
        
        #parse out thse we always need for convenience
        self.pheno_matrix = self.pheno.matrix
        self.sample_ID = self.pheno.row_header.sample_ID[:]
        self.phenotype_ID = self.pheno.col_header.phenotype_ID[:]

        #cache?
        if cache_phenotype:
            self.pheno_matrix = self.pheno_matrix[:]
        
        # Additional pheno col header
        headers = self.pheno.col_header

        #TODO: create pandas.MultiIndex from headers
        child_names = []#headers._v_children.keys()
        child_arrays = []

        for child in headers:
            if child._v_name=="phenotype_ID":
                pass
                #TODO: OS, I removed this as the whole things otherwise crashes if there are no headers
                #continue
            child_names.append(child._v_name)
            child_arrays.append(child[:])
        multiindex = pd.MultiIndex.from_arrays(arrays=child_arrays,names=child_names)
        self.index_frame = pd.DataFrame(data=SP.arange(self.pheno_matrix.shape[1]),index = multiindex)
        self.headers_frame = pd.DataFrame(data=SP.array(child_arrays).T,index=self.phenotype_ID,columns=child_names)

        if 'gene_ID' in headers:
            self.eqtl = True
            self.gene_ID = self.pheno.col_header.gene_ID[:]
            self.gene_pos_start = SP.array([self.pheno.col_header.gene_chrom[:],self.pheno.col_header.gene_start[:]],dtype='int').T
            self.gene_pos_end = SP.array([self.pheno.col_header.gene_chrom[:],self.pheno.col_header.gene_end[:]],dtype='int').T
            self.gene_ID_list= list(set(self.gene_ID))
        else:
            self.eqtl = False
        if 'environment' in headers:
            self.environment  = self.pheno.col_header.environment[:]
            self.environment_list = list(set(self.environment))
        else:
            self.environment = None

        #dimensions
        self.N = self.pheno_matrix.shape[0]
        self.P = self.pheno_matrix.shape[1]

   
    def getPhenotypes(self,phenotype_IDs=None,phenotype_query=None,sample_idx=None,center=True,intersection=False):
        """load Phenotypes
        
        Args:
            phenotype_IDs:      names of phenotypes to load
            phenotype_query:    string hoding a pandas query (e.g. "(environment==1) & (phenotype_ID=='growth')" 
                                selects all phenotypes that have a phenotype_ID equal to growth under environment 1.          
            sample_idx:         Boolean sample index for subsetting individuals
            center:             Boolean: mean center (and mean-fill in missing values if intersection==False)? (default True)
            impute:             imputation of missing values (default: True)
            intersection:       restrict observation to those obseved in all phenotypes? (default: False)
        
        Returns:
            phenotypes:     [N x P] scipy.array of phenotype values for P phenotypes
            sample_idx_intersect:        index of individuals in phenotypes after filtering missing valuesS
        """
        if phenotype_IDs is not None:
            I = SP.array([SP.nonzero(self.phenotype_ID==n)[0][0] for n in phenotype_IDs])
        elif phenotype_query is not None:
            try:
                I = self.index_frame.query(phenotype_query).values[:,0]
            except Exception, arg:
                
                print "query '%s' yielded no results: %s"%phenotype_query, str(arg) 
                                
                I = SP.zeros([0],dtype="int") 
        else:
            I = SP.arange(self.phenotype_ID.shape[0])
        phenotypes = SP.array(self.pheno_matrix[:,I],dtype='float')
        phenotypes = phenotypes[sample_idx]
        Iok = (~SP.isnan(phenotypes))

        if intersection:
            sample_idx_intersect = Iok.all(axis=1)
        else:
            sample_idx_intersect = Iok.any(axis=1)

        phenotypes = phenotypes[sample_idx_intersect]
        Iok = Iok[sample_idx_intersect]

        if center:
            for i in xrange(phenotypes.shape[1]):
                ym = phenotypes[Iok[:,i],i].mean()
                phenotypes[:,i] -= ym
                phenotypes[~Iok[:,i],i] = ym
                phenotypes[:,i] /= phenotypes[:,i].std()
        phenotypes = pd.DataFrame(data=phenotypes, index=self.sample_ID[sample_idx_intersect],columns=self.phenotype_ID[I])
        #calculate overlap of missing values
        return phenotypes, sample_idx_intersect

    def get_pos(self,phenotype_query=None):
        """
        get the positions of the genotypes

        Returns:
            chromosome
            position
            cumulative_position
        """
        assert self.eqtl == True, 'Only for eqtl data'
        if phenotype_query is not None:
            try:
                I = self.index_frame.query(phenotype_query).values[:,0]
            except Exception, arg:
                
                print "query '%s' yielded no results: %s"%phenotype_query, str(arg) 
                                
                I = SP.zeros([0],dtype="int")             
            return {"start" : self.gene_pos_start[I], "end" : self.gene_pos_start[I]}
        else:
            return {"start" : self.gene_pos_start, "end" : self.gene_pos_start}
        

class pheno_reader_h5py_deprecated():
    def __init__(self,file_name):
        self.file_name = file_name
        self.load()

    def load(self,cache_phenotype=True):
        """load data file
        
        Args:
            cache_phenotype:    load phentopyes fully intro memry (default: True)        
        """
        import h5py
        self.f = h5py.File(self.file_name,'r')
        self.pheno = self.f['phenotype']
        
        #parse out thse we always need for convenience
        self.pheno_matrix = self.pheno['matrix']
        self.sample_ID = self.pheno['row_header']['sample_ID'][:]
        self.phenotype_ID = self.pheno['col_header']['phenotype_ID'][:]

        #cache?
        if cache_phenotype:
            self.pheno_matrix = self.pheno_matrix[:]
        
        # Additional pheno col header
        headers = self.pheno['col_header'].keys()
        if 'gene_ID' in headers:
            self.eqtl = True
            self.gene_ID = self.pheno['col_header']['gene_ID'][:]
            self.gene_pos = SP.array([self.pheno['col_header']['gene_chrom'][:],self.pheno['col_header']['gene_start'][:],self.pheno['col_header']['gene_end']],dtype='int').T
            self.gene_ID_list= list(set(self.gene_ID))
        else:
            self.eqtl = False
        if 'environment' in headers:
            self.environment  = self.pheno['col_header/environment'][:]
            self.environment_list = list(set(self.environment))
        else:
            self.environment = None

        #dimensions
        self.N = self.pheno_matrix.shape[0]
        self.P = self.pheno_matrix.shape[1]
        #self.pheno_df = pd.DataFrame(data=self.pheno_matrix,index=self.sample_ID,columns=self.phenotype_ID)

    def getPhenotypes_df(self,sample_idx=None,phenotype_IDs=None,center=True,impute=True,intersection=False):
        """load Phenotypes
        
        Args:
            phenotype_IDs:  names of phenotypes to load
            impute:         imputation of missing values (default: True)
            intersection:   restrict observation to those obseved in all phenotypes? (default: False)
        
        Returns:
            phenotypes:     [N x P] scipy.array of phenotype values for P phenotypes
            sample_idx_intersect:        index of individuals in phenotypes after filtering missing valuesS
        """
        if phenotype_IDs is not None:
            I = SP.array([SP.nonzero(self.phenotype_ID==n)[0][0] for n in phenotype_IDs])
        else:
            I = SP.arange(self.phenotype_ID.shape[0])
        phenotypes = SP.array(self.pheno_matrix[:,I],dtype='float')
        phenotypes = phenotypes[sample_idx]
        Iok = (~SP.isnan(phenotypes))

        if intersection:
            sample_idx_intersect = Iok.all(axis=1)
        else:
            sample_idx_intersect = Iok.any(axis=1)

        phenotypes = phenotypes[sample_idx_intersect]
        Iok = Iok[sample_idx_intersect]

        if center | impute:
            for i in xrange(phenotypes.shape[1]):
                ym = phenotypes[Iok[:,i],i].mean()
                phenotypes[:,i] -= ym
                phenotypes[~Iok[:,i],i] = ym
                phenotypes[:,i] /= phenotypes[:,i].std()
        #calculate overlap of missing values
        return phenotypes, sample_idx_intersect

    def getPhenotypes(self,sample_idx=None,phenotype_IDs=None,center=True,impute=True,intersection=False):
        """load Phenotypes
        
        Args:
            phenotype_IDs:  names of phenotypes to load
            impute:         imputation of missing values (default: True)
            intersection:   restrict observation to those obseved in all phenotypes? (default: False)
        
        Returns:
            phenotypes:     [N x P] scipy.array of phenotype values for P phenotypes
            sample_idx_intersect:        index of individuals in phenotypes after filtering missing valuesS
        """
        if phenotype_IDs is not None:
            I = SP.array([SP.nonzero(self.phenotype_ID==n)[0][0] for n in phenotype_IDs])
        else:
            I = SP.arange(self.phenotype_ID.shape[0])
        phenotypes = SP.array(self.pheno_matrix[:,I],dtype='float')
        phenotypes = phenotypes[sample_idx]
        Iok = (~SP.isnan(phenotypes))

        if intersection:
            sample_idx_intersect = Iok.all(axis=1)
        else:
            sample_idx_intersect = Iok.any(axis=1)

        phenotypes = phenotypes[sample_idx_intersect]
        Iok = Iok[sample_idx_intersect]

        if center | impute:
            for i in xrange(phenotypes.shape[1]):
                ym = phenotypes[Iok[:,i],i].mean()
                phenotypes[:,i] -= ym
                phenotypes[~Iok[:,i],i] = ym
                phenotypes[:,i] /= phenotypes[:,i].std()
        #calculate overlap of missing values
        return phenotypes, sample_idx_intersect
