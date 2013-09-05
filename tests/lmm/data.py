"""Datasets for LMM tests"""
import scipy as SP
import glob
import re
import os

def load(dir):
    dir_name = os.path.dirname(__file__)
    pattern=re.compile('.*/(.*).txt.gz')
    FL = glob.glob(os.path.join(dir_name,dir,'*.txt.gz'))
    RV = {}
    for fn in FL:
        name = pattern.match(fn).group(1)
        RV[name] = SP.loadtxt(fn)
    return RV
    
    
