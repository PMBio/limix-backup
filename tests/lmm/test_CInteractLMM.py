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
        self.dir_name = os.path.dirname(__file__)

    def test_lmm(self):
        """basic main effec tests"""

        for dn in self.datasets:
            D = data.load(os.path.join(self.dir_name,dn))
            N = D['X'].shape[0]
            inter0 = SP.zeros([N,0])#fixed verion: all 0 feature did not work: #inter0 = SP.zeros([N,1])N-by-0 matrix instead of N-by-1 works
            inter1 = SP.ones([N,1])
            lmm = limix.CInteractLMM()        
            lmm.setK(D['K'])
            lmm.setSNPs(D['X'])
            lmm.setCovs(D['Cov'])
            lmm.setPheno(D['Y'])
            #inter0[:]=1
            #inter1[0:N/2]=-1
            lmm.setInter0(inter0)
            lmm.setInter(inter1)
            lmm.process()
            pv = lmm.getPv().ravel()
            D2= SP.sqrt( ((SP.log10(pv)-SP.log10(D['pv']))**2).mean())
            self.assertTrue(D2<1E-6)



    def test_permutation(self):
        #test permutation function
        for dn in self.datasets:
            D = data.load(os.path.join(self.dir_name,dn))
            perm = SP.random.permutation(D['X'].shape[0])
            N = D['X'].shape[0]
            inter0 = SP.zeros([N,0])#fix, as old version with all zero feature does not work#N-by-0 matrix instead of N-by-1 works
            inter1 = SP.ones([N,1])

            #1. set permuattion
            lmm = limix.CInteractLMM()        
            lmm.setInter0(inter0)
            lmm.setInter(inter1)
            lmm.setK(D['K'])
            lmm.setSNPs(D['X'])
            lmm.setCovs(D['Cov'])
            lmm.setPheno(D['Y'])
            if 1:
                #pdb.set_trace()
                perm = SP.array(perm,dtype='int32')#Windows needs int32 as long -> fix interface to accept int64 types
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
            RV = SP.sqrt(D2.mean())
            self.assertTrue(RV<1E-6)


if __name__ == '__main__':
    unittest.main()
