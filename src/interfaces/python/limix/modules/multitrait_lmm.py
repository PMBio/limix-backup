# Copyright(c) 2014, The LIMIX developers (Christoph Lippert, Paolo Francesco Casale, Oliver Stegle)
# All rights reserved.
#
# LIMIX is provided under a 2-clause BSD license.
# See license.txt for the complete license.

import numpy as np
import limix
import limix.modules.varianceDecomposition as vc

class multitrait_lmm(object):
    """description of class"""
    __slots__=["var_decomp","Y"]

    def __init__(self,Y,impute_missing=False, var_decomp=None):
        """
        Args:
            Y:              phenotype matrix [N, P]
            standardize:    if True, impute missing phenotype values by mean value,
                            zero-mean and unit-variance phenotype (Boolean, default False)
        """
        self.var_decomp = variance_decomposition
        self.Y = Y
        pass

