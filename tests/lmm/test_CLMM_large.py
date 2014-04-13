"""LMM testing code"""
import unittest
import scipy as SP
import pdb
import limix
import data
import os


class CLMM_test_large(unittest.TestCase):
    """test class for CLMM"""
    
    def setUp(self):
        self.datasets = ['lmm_data1']
        self.dir_name = os.path.dirname(__file__)

    def test_lmm1(self):
        """basic test, comapring pv"""
        for dn in self.datasets:
            D = data.load(os.path.join(self.dir_name,dn))
            #make vllarg X. This needs to be changed later
            NL = 1000
            self.NL = NL
            X = SP.tile(D['X'],(1,self.NL))
            lmm = limix.CLMM()
            lmm.setK(D['K'])
            lmm.setSNPs(X)
            lmm.setCovs(D['Cov'])
            lmm.setPheno(D['Y'])
            lmm.process()
            pv = lmm.getPv().ravel()
            BetaSte = lmm.getBetaSNPste().ravel()
            Beta = lmm.getBetaSNP()
            D2pv= (SP.log10(pv)-SP.log10(SP.tile(D['pv'],self.NL))**2)
            D2Beta= (Beta-SP.tile(D['Beta'],self.NL))**2
            D2BetaSte = (BetaSte-SP.tile(D['BetaSte'],self.NL))**2
            RV = SP.sqrt(D2pv.mean())<1E-6
            RV = RV & (D2Beta.mean()<1E-6)
            RV = RV & (D2BetaSte.mean()<1E-6)
            self.assertTrue(RV)



if __name__ == '__main__':
    unittest.main()
