"""LMM testing code"""
import unittest
import scipy as SP
import pdb
import limix
import data
import os


class CKroneckerLMM_test(unittest.TestCase):
    """test class for CLMM"""
    
    def setUp(self):
        self.datasets = ['lmm_data1']
        self.dir_name = os.path.dirname(__file__)

    def test_lmm(self):
        """basic test, comparing pv to a standard LMM equivalent"""
        for dn in self.datasets:
            D = data.load(os.path.join(self.dir_name,dn))
            #construct Kronecker LMM model which has the special case of standard LMM
            #covar1: genotype matrix
            K1r = D['K']
            K1c = SP.eye(1)
            K2r = SP.eye(D['K'].shape[0])
            K2c = SP.eye(1)
            A   = SP.eye(1)
            Acov = SP.eye(1)
            Xcov  = D['Cov'][:,SP.newaxis]
            X      = D['X']
            Y      = D['Y'][:,SP.newaxis]
                        
            lmm = limix.CKroneckerLMM()
            lmm.setK1r(K1r)
            lmm.setK1c(K1c)
            lmm.setK2r(K2r)
            lmm.setK2c(K2c)
            
            lmm.setSNPs(X)
            #add covariates
            lmm.addCovariates(Xcov,Acov)
            #add SNP design
            lmm.setSNPcoldesign(A)
            lmm.setPheno(Y)
            lmm.setNumIntervalsAlt(0)
            lmm.setNumIntervals0(100)
            
            lmm.process()
            pv = lmm.getPv().ravel()
            D2= ((SP.log10(pv)-SP.log10(D['pv']))**2)
            RV = SP.sqrt(D2.mean())
            #print "\n"
            #print pv[0:10]
            #print D['pv'][0:10]
            #print RV
            #pdb.set_trace()
            self.assertTrue(RV<1E-6)

    def test_lmm2(self):
        """another test, establishing an lmm-equivalent by a design matrix choice"""
        for dn in self.datasets:
            D = data.load(os.path.join(self.dir_name,dn))
            #construct Kronecker LMM model which has the special case of standard LMM
            #covar1: genotype matrix
            N = D['K'].shape[0]
            P = 3
            K1r = D['K']
            #K1c = SP.zeros([2,2])
            #K1c[0,0] = 1
            K1c = SP.eye(P)
            K2r = SP.eye(N)
            K2c = SP.eye(P)

            #A   = SP.zeros([1,2])
            #A[0,0] =1
            A = SP.eye(P)
            Acov = SP.eye(P)
            Xcov = D['Cov'][:,SP.newaxis]
            X      = D['X']
            Y      = D['Y'][:,SP.newaxis]
            Y      = SP.tile(Y,(1,P))
                        
            lmm = limix.CKroneckerLMM()
            lmm.setK1r(K1r)
            lmm.setK1c(K1c)
            lmm.setK2r(K2r)
            lmm.setK2c(K2c)
            
            lmm.setSNPs(X)
            #add covariates
            lmm.addCovariates(Xcov,Acov)
            #add SNP design
            lmm.setSNPcoldesign(A)
            lmm.setPheno(Y)
            lmm.setNumIntervalsAlt(0)
            lmm.setNumIntervals0(100)
            
            lmm.process()
            
            #get p-values with P-dof:
            pv_Pdof = lmm.getPv().ravel()
            #transform in P-values with a single DOF:
            import scipy.stats as st
            lrt = st.chi2.isf(pv_Pdof,P)/P
            pv = st.chi2.sf(lrt,1)
            #compare with single DOF P-values:
            D2= ((SP.log10(pv)-SP.log10(D['pv']))**2)
            RV = SP.sqrt(D2.mean())
            #print "\n"
            #print pv[0:10]
            #print D['pv'][0:10]
            #print RV
            #pdb.set_trace()
            self.assertTrue(RV<1E-6)



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
