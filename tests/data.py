"""Datasets for LMM tests"""
import scipy as SP
import glob
import re
import os

def load(dir):
    pattern=re.compile('.*/(.*).txt.gz')
    FL = glob.glob(os.path.join(dir,'*.txt.gz'))
    RV = {}
    for fn in FL:
        name = pattern.match(fn).group(1)
        RV[name] = SP.loadtxt(fn)
    return RV

def dump(R,dir):
    for r in R.keys():
        fn = os.path.join(dir,r+'.txt.gz')
        SP.savetxt(fn,R[r])
    
    
