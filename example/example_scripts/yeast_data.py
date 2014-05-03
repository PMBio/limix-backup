import limix.modules.data_new as DATA
import limix.modules.genotype_reader as gr
import limix.modules.phenotype_reader as phr
import scipy as SP
file_name = './../../tutorials/data/smith_2008/smith08.hdf5'


import tables
f = tables.openFile(file_name,'r')
pheno = f.root.phenotype
        
        
# Additional pheno col header
headers = pheno.col_header

geno_reader  = gr.genotype_reader_tables(file_name)
pheno_reader = phr.pheno_reader_tables(file_name)

data = DATA.QTLData(geno_reader=geno_reader,pheno_reader=pheno_reader)
