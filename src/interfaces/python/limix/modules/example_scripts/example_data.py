# Copyright(c) 2014, The LIMIX developers (Christoph Lippert, Paolo Francesco Casale, Oliver Stegle)
# All rights reserved.
#
# LIMIX is provided under a 2-clause BSD license.
# See license.txt for the complete license.

import limix.modules.data as DATA
import scipy as SP

data_file = "./example_data/kruglyak.hdf5"
pheno_names_complete = SP.array(['Calcium_Chloride', 'Cisplatin', 'Cobalt_Chloride', 'Congo_red',
       'Copper', 'Cycloheximide', 'Diamide', 'E6_Berbamine', 'Ethanol',
       'Formamide', 'Galactose', 'Hydrogen_Peroxide', 'Hydroxyurea',
       'Indoleacetic_Acid', 'Lactate', 'Lactose', 'Lithium_Chloride',
       'Magnesium_Chloride', 'Magnesium_Sulfate', 'Maltose', 'Mannose',
       'Menadione', 'Neomycin', 'Paraquat', 'Raffinose', 'Tunicamycin',
       'x4-Hydroxybenzaldehyde', 'x4NQO', 'x5-Fluorocytosine',
       'x5-Fluorouracil', 'x6-Azauracil', 'Xylose', 'YNB', 'YNB:ph3',
       'YNB:ph8', 'YPD:15C', 'YPD:37C', 'YPD:4C'])
data = DATA.QTLData(data_file)
