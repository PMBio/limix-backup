# Copyright(c) 2014, The LIMIX developers (Christoph Lippert, Paolo Francesco Casale, Oliver Stegle)
# All rights reserved.
#
# LIMIX is provided under a 2-clause BSD license.
# See license.txt for the complete license.


"""
Gaussian Process Package
========================

Holds all Gaussian Process classes, which hold all informations for a Gaussian Process to work porperly.

.. class **GP**: basic class for GP regression:
   * claculation of log marginal likelihood
   * prediction
   * data rescaling
   * transformation into log space

   
"""

try:
    __import__('pkg_resources').declare_namespace(__name__)
except ImportError:
    from pkgutil import extend_path
    __path__ = extend_path(__path__, __name__)


#Default: import gp_base
from pygp.gp.gp_base import *
