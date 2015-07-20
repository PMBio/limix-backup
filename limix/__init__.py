from mtSet import MTSet
from varDecomp import VarianceDecomposition 

#verbose flags for the entire module
verbose = False

def getVerbose(verbose):
    """resolve verbose flag, using module settings if verbose=None"""
    if verbose is None:
        verbose = limix.verbose
    else:
        verbose = verbose
    return verbose

def test():
	from unittest import TestLoader
	from unittest import TextTestRunner
	import os
	folder = os.path.dirname(os.path.realpath(__file__))
	suite = TestLoader().discover(folder)
	TextTestRunner(verbosity=2).run(suite)
