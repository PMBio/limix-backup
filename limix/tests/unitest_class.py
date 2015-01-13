import sys
import os
path_abs = os.path.dirname(os.path.abspath(sys.argv[0]))
path_mtSet = os.path.join(path_abs,'../..')
sys.path.append(path_mtSet)
import unittest
import scipy as SP
import scipy.linalg as LA
import pdb

class unittestClass(unittest.TestCase):
    """test class for optimization""" 
    
    def saveStuff(self,fbasename,ext):
        """ util function """ 
        base = './data/res_'+fbasename+'_'
        for key in ext.keys(): 
            SP.savetxt(base+key+'.txt',ext[key])

    def loadStuff(self,fbasename,keys):
        """ util function """ 
        RV = {}
        base = './data/res_'+fbasename+'_'
        for key in keys: 
            RV[key] = SP.loadtxt(base+key+'.txt')
        return RV

    def assess(self,fbasename,ext):
        """ returns a bool vector """
        real = self.loadStuff(fbasename,ext.keys()) 
        RV = SP.all([((ext[key]-real[key])**2).mean()<1e-6 for key in ext.keys()])
        return RV

