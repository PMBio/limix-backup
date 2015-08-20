import limix
from mtSet import MTSet


def test():
	from unittest import TestLoader
	from unittest import TextTestRunner
	import os
	folder = os.path.dirname(os.path.realpath(__file__))
	suite = TestLoader().discover(folder)
	TextTestRunner(verbosity=2).run(suite)

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
