# Copyright(c) 2014, The LIMIX developers (Christoph Lippert, Paolo Francesco Casale, Oliver Stegle)
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

__all__ = ['']

__version__ = '0.6.5'

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
