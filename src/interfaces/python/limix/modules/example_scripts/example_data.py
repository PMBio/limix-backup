# Copyright(c) 2014, The LIMIX developers (Christoph Lippert, Paolo Francesco Casale, Oliver Stegle)
# All rights reserved.
#
# LIMIX is provided under a 2-clause BSD license.
# See license.txt for the complete license.

import limix.modules.data as DATA
import scipy as SP

data_file = "./example_data/kruglyak.hdf5"

data = DATA.QTLData(data_file)
