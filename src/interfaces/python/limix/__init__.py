__all__ = ['']

#make core available on flat import
import core
from core import *
#import limix
import limix
#verbose flags for the entire module
verbose = False


def getVerbose(verbose):
	"""resolve verbose flag, using module settings if verbose=None"""
	if verbose is None:
		verbose = limix.verbose
	else:
		verbose = verbose
	return verbose