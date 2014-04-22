import scipy as SP
import data_util as du

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
        if 'gene_ID' in headers:
            self.eqtl = True
            self.gene_ID = self.pheno.col_header.gene_ID[:]
            self.gene_pos = SP.array([self.pheno.col_header.gene_chrom[:],self.pheno.col_header.gene_start[:],self.pheno.col_header.gene_end],dtype='int').T
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

class pheno_reader_h5py():
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