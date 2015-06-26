import unittest
import limix 
import scipy as sp
import scipy.linalg as LA
import pdb

class TestMTSet(unittest.TestCase):
    """test class for optimization""" 
    
    def setUp(self):
        sp.random.seed(0)
        self.Y = sp.loadtxt('./data/Y.txt') 
        self.XX = sp.loadtxt('./data/XX.txt') 
        self.Xr = sp.loadtxt('./data/Xr.txt') 
        self.N,self.P = self.Y.shape
        self.write = False 

    def test_mtSetNull(self):
        fbasename = 'mtSetNull'
        setTest = limix.MTSet(Y=self.Y, R=self.XX)
        nullMTInfo = setTest.fitNull(cache=False)
        ext = {'Cg':nullMTInfo['Cg'],'Cn':nullMTInfo['Cn']}
        if self.write: self.saveStuff(fbasename,ext)
        RV = self.assess(fbasename,ext)
        self.assertTrue(RV)

    def test_mtSetNull_eigenCache(self):
        fbasename = 'mtSetNull'
        S,U = LA.eigh(self.XX)
        setTest = limix.MTSet(Y=self.Y, S_R=S, U_R=U)
        nullMTInfo = setTest.fitNull(cache=False)
        ext = {'Cg':nullMTInfo['Cg'],'Cn':nullMTInfo['Cn']}
        if self.write: self.saveStuff(fbasename,ext)
        RV = self.assess(fbasename,ext)
        self.assertTrue(RV)

    def test_mtSet_base(self):
        fbasename = 'mtSet_base'
        setTest = limix.MTSet(self.Y, R= self.XX)
        optInfo = setTest.optimize(self.Xr)
        ext = {'Cr':optInfo['Cr'],
               'Cg':optInfo['Cg'],
               'Cn':optInfo['Cn']}
        if self.write: self.saveStuff(fbasename,ext)
        RV = self.assess(fbasename,ext)
        self.assertTrue(RV)

    def test_mtSet_eigenCache(self):
        fbasename = 'mtSet_base'
        S,U = LA.eigh(self.XX)
        setTest = limix.MTSet(Y=self.Y, S_R=S, U_R=U)
        optInfo = setTest.optimize(self.Xr)
        ext = {'Cr':optInfo['Cr'],
               'Cg':optInfo['Cg'],
               'Cn':optInfo['Cn']}
        RV = self.assess(fbasename,ext)
        self.assertTrue(RV)

    def saveStuff(self,fbasename,ext):
        """ util function """ 
        base = './data/res_'+fbasename+'_'
        for key in ext.keys(): 
            sp.savetxt(base+key+'.txt',ext[key])

    def loadStuff(self,fbasename,keys):
        """ util function """ 
        RV = {}
        base = './data/res_'+fbasename+'_'
        for key in keys: 
            RV[key] = sp.loadtxt(base+key+'.txt')
        return RV

    def assess(self,fbasename,ext):
        """ returns a bool vector """
        real = self.loadStuff(fbasename,ext.keys()) 
        RV = sp.all([((ext[key]-real[key])**2).mean()<1e-6 for key in ext.keys()])
        return RV

if 0:

    def test_mtSetPCnull_base(self):
        fbasename = 'mtSetPCnull_base'
        setTest = MTST.MultiTraitSetTest(self.Y,XX=None)
        nullMTInfo = setTest.fitNull(cache=False)
        ext = {'Cn':nullMTInfo['Cn']}
        if self.write: self.saveStuff(fbasename,ext)
        RV = self.assess(fbasename,ext)
        self.assertTrue(RV)

    def test_mtSetPCnull_fixed(self):
        fbasename = 'mtSetPCnull_fixed'
        setTest = MTST.MultiTraitSetTest(self.Y,XX=None,F=self.Xr)
        nullMTInfo = setTest.fitNull(cache=False)
        ext = {'Cg':nullMTInfo['Cg'],'Cn':nullMTInfo['Cn']}
        if self.write: self.saveStuff(fbasename,ext)
        RV = self.assess(fbasename,ext)
        self.assertTrue(RV)

    def test_mtSetPC_base(self):
        fbasename = 'mtSetPC_base'
        setTest = MTST.MultiTraitSetTest(self.Y,XX=None)
        optInfo = setTest.optimize(self.Xr)
        ext = {'Cr':optInfo['Cr'],
               'Cn':optInfo['Cn']}
        if self.write: self.saveStuff(fbasename,ext)
        RV = self.assess(fbasename,ext)
        self.assertTrue(RV)

    def test_mtSetPC_fixed(self):
        fbasename = 'mtSetPC_fixed'
        setTest = MTST.MultiTraitSetTest(self.Y,XX=None,F=self.Xr[:,:2])
        optInfo = setTest.optimize(self.Xr)
        ext = {'Cr':optInfo['Cr'],
               'Cn':optInfo['Cn']}
        if self.write: self.saveStuff(fbasename,ext)
        RV = self.assess(fbasename,ext)
        self.assertTrue(RV)


if __name__ == '__main__':
    unittest.main()

