"""run all tests"""

import unittest
import glob
import sys

if __name__ == '__main__':
    if len(sys.argv)>1:
        #1. add LIMIX path to environment
        limix_path = sys.argv[1]
        sys.path.insert(0,limix_path)

    if len(sys.argv)>2:
	folders = [sys.argv[2]]
    else:
    	folders = set(glob.glob('*'))-set(glob.glob('*.*'))

    # Gather all tests in suite
    suite = unittest.TestSuite()
    for folder in folders:
        tests = unittest.TestLoader().discover(folder,'test*')
        suite.addTests(tests)

    # run all tests
    unittest.TextTestRunner(verbosity=2).run(suite)
