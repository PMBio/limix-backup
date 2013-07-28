"""LMM testing code"""
import scipy as SP
import pdb
import sys
import limix

from helper.helper import message

class CLMM_test:
    """test class for CLMM"""
    
    def __init__(self):
        self.generate()

    def generate(self):
        SP.random.seed(1)
        self.N = 100
        self.S = 10
    
        self.X = SP.random.randn(self.N,self.S)
        self.Y = SP.dot(self.X,SP.random.randn(self.S,1))
        self.Y += SP.random.randn(self.N,1)
        self.C = SP.ones([self.N,1])
        self.K = SP.eye(self.N)
        self.pv_true = SP.array([  2.51035504e-01,   7.87110691e-11,   4.97690005e-01,4.64168307e-01,   1.35978541e-03,   5.78090595e-01, 5.79741538e-02,   4.27432447e-04,   3.35524125e-01,3.94149107e-06])
        

    def test_lmm1(self):
        """basic test, comapring pv"""
        lmm = limix.CLMM()
        lmm.setK(self.K)
        lmm.setSNPs(self.X)
        lmm.setCovs(self.C)
        lmm.setPheno(self.Y)
        lmm.process()
        pv = lmm.getPv().ravel()
        D2= ((SP.log10(pv)-SP.log10(self.pv_true))**2)
        RV = SP.sqrt(D2.mean())<1E-6
        print '   ...pvalue %s' % message(RV)


    def test_permutation(self):
        #test permutation function
        perm = SP.random.permutation(self.X.shape[0])

        #1. set permuattion
        lmm = limix.CLMM()
        lmm.setK(self.K)
        lmm.setSNPs(self.X)
        lmm.setCovs(self.C)
        lmm.setPheno(self.Y)
        lmm.setPermutation(perm)
        lmm.process()
        pv_perm1 = lmm.getPv().ravel()
        #2. do by hand
        lmm = limix.CLMM()
        lmm.setK(self.K)
        lmm.setSNPs(self.X[perm])
        lmm.setCovs(self.C)
        lmm.setPheno(self.Y)
        lmm.process()
        pv_perm2 = lmm.getPv().ravel()
        D2 = (SP.log10(pv_perm1)-SP.log10(pv_perm2))**2
        RV = SP.sqrt(D2.mean())<1E-6
        print '   ...permutation %s' % message(RV)
    
    def test_all(self):
        print '... testing CLMM'
        #test basic LMM
        self.test_lmm1()
        #test permutation
        self.test_permutation()


class CInteractLMM_test:
    """Interaction test"""
    def __init__(self):
        pass

    def test_all(self):
        RV = False
        print 'CInteractLMM IMPLEMENTED %s' % message(RV)


if __name__ == '__main__':
    RV = True
    testlmm = CLMM_test();
    testlmm.test_all()
    testilmm = CInteractLMM_test()
    testilmm.test_all()
