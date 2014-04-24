# Copyright(c) 2014, The LIMIX developers (Christoph Lippert, Paolo Francesco Casale, Oliver Stegle)
# All rights reserved.
#
# LIMIX is provided under a 2-clause BSD license.
# See license.txt for the complete license.

import data_new as DATA
import genotype_reader as gr
import phenotype_reader as phr
import scipy as SP

data_file = "./example_data/kruglyak.hdf5"

geno_reader  = gr.genotype_reader_tables(data_file)
pheno_reader = phr.pheno_reader_tables(data_file)

#geno_reader  = gr.genotype_reader_h5py(data_file)
#pheno_reader = phr.pheno_reader_h5py(data_file)

data = DATA.QTLData(geno_reader=geno_reader,pheno_reader=pheno_reader)
