"""run all tests"""

import scipy as SP
import glob
import sys
import os
sys.path.append('covar')
sys.path.append('helper')

if __name__ == '__main__':
    if 1:
        #1. add LIMIX path to environment
        limix_path = sys.argv[1]
        #is PYTHONPATH defined?
        sys.path.append(limix_path)
    if 0:
        if 'PYTHONPATH' in os.environ.keys():
            os.environ['PYTHONPATH'] += ':'+ limix_path
        else:
            os.environ['PYTHONPATH'] = limix_path

    # get tests to run:
    FL=glob.glob('./*/test_*.py')

    #run them

    for fn in FL:
        print "running test %s"  % (os.path.basename(fn))
        rr = execfile(fn)
        pass

        
    
