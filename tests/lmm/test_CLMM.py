"""LMM testing code"""
import unittest
import scipy as SP
import pdb
import limix
import data
import os
import sys


class CLMM_test(unittest.TestCase):
    """test class for CLMM"""
    
    def setUp(self):
        self.datasets = ['lmm_data1']
        self.dir_name = os.path.dirname(__file__)

    def test_lmm1(self):
        """basic test, comapring pv"""
        for dn in self.datasets:
            D = data.load(os.path.join(self.dir_name,dn))
            lmm = limix.CLMM()
            lmm.setK(D['K'])
            lmm.setSNPs(D['X'])
            lmm.setCovs(D['Cov'])
            lmm.setPheno(D['Y'])
            lmm.process()
            pv = lmm.getPv().ravel()
            BetaSte = lmm.getBetaSNPste().ravel()
            Beta = lmm.getBetaSNP()
            D2pv= ((SP.log10(pv)-SP.log10(D['pv']))**2)
            D2Beta= (Beta-D['Beta'])**2
            D2BetaSte = (BetaSte-D['BetaSte'])**2
            RV = SP.sqrt(D2pv.mean())<1E-6
            RV = RV & (D2Beta.mean()<1E-6)
            RV = RV & (D2BetaSte.mean()<1E-6)
            self.assertTrue(RV)

    def test_exceptions(self):
        D = data.load(os.path.join(self.dir_name,self.datasets[0]))
        lmm  = limix.CLMM()
        N = 100
        K = SP.eye(N)
        X = SP.randn(N,100)
        Y = SP.randn(N+1,1)
        Cov = SP.randn(N,1)
        lmm.setK(K)
        lmm.setSNPs(X)
        lmm.setCovs(Cov)
        lmm.setPheno(Y)
        try:
            lmm.process()
        except Exception,e:
            self.assertTrue(1==1)
            pass

    def test_permutation(self):
        #test permutation function
        for dn in self.datasets:
            D = data.load(os.path.join(self.dir_name,dn))
            perm = SP.random.permutation(D['X'].shape[0])
            #1. set permuattion
            lmm = limix.CLMM()
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
            lmm = limix.CLMM()
            lmm.setK(D['K'])
            lmm.setSNPs(D['X'][perm])
            lmm.setCovs(D['Cov'])
            lmm.setPheno(D['Y'])
            lmm.process()
            pv_perm2 = lmm.getPv().ravel()
            D2 = (SP.log10(pv_perm1)-SP.log10(pv_perm2))**2
            RV = SP.sqrt(D2.mean())
            self.assertTrue(RV<1E-6)



class CInteractLMM_test:
    """Interaction test"""
    def __init__(self):
        pass

    def test_all(self):
        RV = False
        #print 'CInteractLMM IMPLEMENTED %s' % message(RV)


if __name__ == '__main__':
    unittest.main()
