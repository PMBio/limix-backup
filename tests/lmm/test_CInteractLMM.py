"""LMM testing code"""
import unittest
import scipy as SP
import pdb
import limix
import data
import os

class CIntearctLMM_test(unittest.TestCase):
    """
    test class for CIntearctLMM
    Status: currenlty only testing the special case of a CinteractLMM that is equilvanet to LMM
    """
    
    def setUp(self):
        self.datasets = ['lmm_data1']

    def test_lmm(self):
        """basic main effec tests"""

        for dn in self.datasets:
            D = data.load(dn)

            N = D['X'].shape[0]
            inter0 = SP.zeros([N,1])
            inter1 = SP.ones([N,1])
            lmm = limix.CInteractLMM()        
            lmm.setK(D['K'])
            lmm.setSNPs(D['X'])
            lmm.setCovs(D['Cov'])
            lmm.setPheno(D['Y'])
            lmm.setInter0(inter0)
            lmm.setInter(inter1)
            lmm.process()
            pv = lmm.getPv().ravel()
            D2= SP.sqrt( ((SP.log10(pv)-SP.log10(D['pv']))**2).mean())
            self.assertTrue(D2<1E-6)



    def test_permutation(self):
        #test permutation function
        for dn in self.datasets:
            D = data.load(dn)
            perm = SP.random.permutation(D['X'].shape[0])

            D = data.load(dn)
            N = D['X'].shape[0]
            inter0 = SP.zeros([N,1])
            inter1 = SP.ones([N,1])

            #1. set permuattion
            lmm = limix.CInteractLMM()        
            lmm.setInter0(inter0)
            lmm.setInter(inter1)
            lmm.setK(D['K'])
            lmm.setSNPs(D['X'])
            lmm.setCovs(D['Cov'])
            lmm.setPheno(D['Y'])
            lmm.setPermutation(perm)
            lmm.process()
            pv_perm1 = lmm.getPv().ravel()
            #2. do by hand
            lmm = limix.CInteractLMM()        
            lmm.setInter0(inter0)
            lmm.setInter(inter1)
            lmm.setK(D['K'])
            lmm.setSNPs(D['X'][perm])
            lmm.setCovs(D['Cov'])
            lmm.setPheno(D['Y'])
            lmm.process()
            pv_perm2 = lmm.getPv().ravel()
            D2 = (SP.log10(pv_perm1)-SP.log10(pv_perm2))**2
            RV = SP.sqrt(D2.mean())<1E-6
            self.assertTrue(RV)


if __name__ == '__main__':
    unittest.main()
