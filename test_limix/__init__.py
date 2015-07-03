def test():
	from unittest import TestLoader
	from unittest import TextTestRunner
	import os
	folder = os.path.dirname(os.path.realpath(__file__))
	suite = TestLoader().discover(folder)
	TextTestRunner(verbosity=2).run(suite)
