# Copyright(c) 2014, The LIMIX developers (Christoph Lippert, Paolo Francesco Casale, Oliver Stegle)
# All rights reserved.
#
# LIMIX is provided under a 2-clause BSD license.
# See license.txt for the complete license.

import limix.modules.qtl as QTL
import scipy as SP

pheno_names_complete = SP.array(['Calcium_Chloride', 'Cisplatin', 'Cobalt_Chloride', 'Congo_red',
       'Copper', 'Cycloheximide', 'Diamide', 'E6_Berbamine', 'Ethanol',
       'Formamide', 'Galactose', 'Hydrogen_Peroxide', 'Hydroxyurea',
       'Indoleacetic_Acid', 'Lactate', 'Lactose', 'Lithium_Chloride',
       'Magnesium_Chloride', 'Magnesium_Sulfate', 'Maltose', 'Mannose',
       'Menadione', 'Neomycin', 'Paraquat', 'Raffinose', 'Tunicamycin',
       'x4-Hydroxybenzaldehyde', 'x4NQO', 'x5-Fluorocytosine',
       'x5-Fluorouracil', 'x6-Azauracil', 'Xylose', 'YNB', 'YNB:ph3',
       'YNB:ph8', 'YPD:15C', 'YPD:37C', 'YPD:4C'])



X = data.getGenotypes()
K = data.getCovariance()
pos = data.getPos()
#individual LMM scan
[Y,I] = data.getPhenotypes(phenotype_IDs=pheno_names_complete,intersection=True)
X = X[I]
K = K[I,:][:,I]
result = QTL.simple_interaction_kronecker(snps=X,phenos=Y[:,0:2],covs = None,Acovs=None,Asnps1=SP.eye(2),Asnps0=SP.ones((1,2)),K1r=K,K2r=SP.eye(K.shape[1]),K1c=None,K2c=None,covar_type='lowrank_diag',rank=1,searchDelta=False)