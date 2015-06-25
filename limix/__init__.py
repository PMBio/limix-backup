#from modules2 import MTSet

#import self
import limix

#import a selection of key deprecated modules
import deprecated.modules as modules
# from modules2 import *

#verbose flags for the entire module
verbose = False

def getVerbose(verbose):
    """resolve verbose flag, using module settings if verbose=None"""
    if verbose is None:
        verbose = limix.verbose
    else:
        verbose = verbose
    return verbose
