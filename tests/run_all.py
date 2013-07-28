"""run all tests"""

import unittest
import glob

if __name__ == '__main__':

    # list of folders containing tests
    folders = set(glob.glob('*'))-set(glob.glob('*.*'))

    # Gather all tests in suite
    suite = unittest.TestSuite()
    for folder in folders:
        tests = unittest.TestLoader().discover(folder,'test*')
        suite.addTests(tests)

    # run all tests
    unittest.TextTestRunner(verbosity=2).run(suite)
