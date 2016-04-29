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

#make core available on flat import
from .core import *
#verbose flags for the entire module
_verbose = False


def getVerbose(verbose=None):
	"""resolve verbose flag, using module settings if verbose=None"""
	global _verbose
	if verbose is None:
		return _verbose
	_verbose = verbose
	return verbose
