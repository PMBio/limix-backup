import scipy as SP
import h5py
import data_util as du

class pheno_reader_hdf5():
    def __init__(self,file_name):
        self.file_name = file_name
        self.load()

    def load(self,cache_phenotype=True):
        """load data file
        
        Args:
            cache_phenotype:    load phentopyes fully intro memry (default: True)        
        """
        self.f = h5py.File(self.file_name,'r')
        self.pheno = self.f['phenotype']
        #TODO: load all row and column headers for genotype and phenotype
        
        #parse out thse we alwasy need for convenience
        self.phenoM = self.pheno['matrix']
        self.sample_ID = self.pheno['row_header']['sample_ID'][:]
        self.phenotype_ID = self.pheno['col_header']['phenotype_ID'][:]

        #cache?
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
        self.N = self.phenoM.shape[0]
        self.P = self.phenoM.shape[1]

    def getPhenotypes(self,i0=None,i1=None,phenotype_IDs=None,center=True,impute=True,intersection=False):
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