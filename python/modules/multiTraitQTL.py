"""testing script for multi trait covariance stuff"""
import sys
sys.path.append('./..')
sys.path.append('./../../..')

import scipy as SP
import scipy.linalg
import limix as mmtk
import time
import pdb
import pylab as PL


def scale_k(k, verbose=False):
    c = SP.sum((SP.eye(len(k)) - (1.0 / len(k)) * SP.ones(k.shape)) * SP.array(k))
    scalar = (len(k) - 1) / c
    if verbose:
        print 'Kinship scaled by: %0.4f' % scalar
    k = scalar * k
    return k



class CMMT(object):
    def __init__(self,X=None,E=None,Y=None,C=None,Kpop=None,Kgeno=None,T=2,standardize=True,Ve00=SP.nan,Ve11=SP.nan,Ve01=SP.nan,Vg00=SP.nan,Vg01=SP.nan,Vg11=SP.nan):
        """
        X: SNPS [N x S]
        Y: phenotype [N x 1]
        E: env. indocator [N x 1]
        C: covariates [N x C]
        T: number of traits (2)
        Kpop[XX^T]: population covariance [N x N]
        Kgeno: genotype identity covariance [N x N]
        standardize[True]: standardize phenotype per trait?
        """
        assert T==2, 'supporting 2 env. only now'
        assert Y.shape[1]==1, 'Y needs to be [N x 1]'
        assert X.shape[0]==Y.shape[0], 'X and Y have incompatible shape'
        assert E.shape[0]==Y.shape[0], 'E and Y have incompatible shape'
        assert Kpop.shape[0]==Y.shape[0], 'Kpop and Y have incompatible shape'
        assert Kgeno.shape[0]==Y.shape[0], 'Kgeno and Y have incompatible shape'
        assert len(SP.unique(E))==2, 'Only supporting 2 environments (0/1) right now'
        assert 0 in E, 'Only supporting 2 environments (0/1) right now'
        assert 1 in E, 'Only supporting 2 environments (0/1) right now'


        if C is None:
            C = SP.ones([Y.shape[0],1])
        if Kpop is None:
            Kpop = 1.0/X.shpape[1] * SP.dot(X,X.T)
            
        self.X = X
        self.E = E
        self.Y = Y
        self.T =T
        self.C = C
        self.Kpop = scale_k(Kpop)
        self.Kgeno= scale_k(Kgeno)
        
        #variance component init:
        self.VinitE = SP.array([[Ve00,Ve01],[Ve01,Ve11]])
        self.VinitG = SP.array([[Vg00,Vg01],[Vg01,Vg11]])
        
        #standardize data
        if standardize:
            self._standardize()

        #init lmm object: testing
        self.lmm =  mmtk.CLMM()
        self.lmi = mmtk.CInteractLMM()
        
        #fit variance components using delta - fitting
        if (SP.isnan(self.VinitG[0,0])):
            self._init_variance()
         
        
    
    def _standardize(self):
        """standardize expression per environment"""
        for t in xrange(self.T):
            Ie = (self.E[:,0]==t)
            _Y = self.Y[Ie]
            _Y-=_Y.mean()
            _Y/=_Y.std()
            self.Y[Ie] = _Y
        pass
    
    def _init_variance(self):
        """guess variance components based on independent analysis"""
        D =[]
        for t in xrange(self.T):
            Ie = (self.E[:,0]==t)
            _Y = self.Y[Ie]
            _C = self.C[Ie]
            _X = SP.zeros([_C.shape[0],0])
            self.lmm.setK(self.Kpop[Ie,:][:,Ie])
            self.lmm.setSNPs(_X)
            self.lmm.setCovs(_C)
            self.lmm.setPheno(_Y)
            #optimize delta -10..10
            self.lmm.setVarcompApprox0(-20,10,1000)
            self.lmm.process()
            #get delta
            ldelta = self.lmm.getLdelta0().flatten()
            D.append(SP.exp(ldelta))
            pass
        #initialize variances from delta fit
        #note we assume that Sg^2 + Se^2 =1, i.e. each trait has been standardized
        self.VinitG[0,0] = 1./(1+D[0])
        self.VinitG[1,1] = 1./(1+D[1])
        self.VinitE[0,0] = D[0]/(1+D[0])
        self.VinitE[1,1] = D[1]/(1+D[1])
        
        
    def _init_GP(self):
        """create GP instance for fitting"""
        GP = {}
        #overall covariace
        GP['covar'] = mmtk.CSumCF()
        
        #1. genotype X env covariance
        GP['covar_GG'] = mmtk.CFixedCF(self.Kpop)
        #freeform covariance: requiring number of traits/group (T)
        GP['covar_GE'] = mmtk.CCovFreeform(self.T)
        GP['covar_G'] = mmtk.CProductCF()
        GP['covar_G'].addCovariance(GP['covar_GG'])
        GP['covar_G'].addCovariance(GP['covar_GE'])
        
        #2. env covariance:
        GP['covar_EG'] = mmtk.CFixedCF(self.Kgeno)
        #freeform covariance: requiring number of traits/group (T)
        GP['covar_EE'] = mmtk.CCovFreeform(self.T)
        GP['covar_E'] = mmtk.CProductCF()
        GP['covar_E'].addCovariance(GP['covar_EG'])
        GP['covar_E'].addCovariance(GP['covar_EE'])      
          
        #add to sum CF
        GP['covar'].addCovariance(GP['covar_G'])
        GP['covar'].addCovariance(GP['covar_E'])
          
        #liklihood: NULL likleihhod; all variance is explained by covariance function.
        GP['ll'] = mmtk.CLikNormalNULL()
        GP['data'] =  mmtk.CData()
        GP['hyperparams'] = mmtk.CGPHyperParams()
        #Create GP instance
        GP['gp']=mmtk.CGPbase(GP['data'],GP['covar'],GP['ll'])
        #set data
        GP['gp'].setY(self.Y)
        #create X: we have 2 covariances that need inputs (Cover_EE,covar_GE)
        Xgp = SP.concatenate((self.E,self.E),axis=1)
        GP['gp'].setX(Xgp)
        #optimization interface
        GP['gpopt'] = mmtk.CGPopt(GP['gp'])
        #filter?
        covar_mask = SP.ones([GP['covar'].getNumberParams(),1])
        covar_mask[0] = 0 
        covar_mask[4] = 0 
        mask = mmtk.CGPHyperParams()
        mask['covar'] = covar_mask
        GP['gpopt'].setParamMask(mask)
        #hyperparams object
        GP['hyperparams'] = mmtk.CGPHyperParams()
        self.GP=GP
    
    
    def _var2params(self,V,vr=1E-4):
        """
        Create param object from variance component array
        Missing values will be replaced by random values
        """
        while 1==1:
            if SP.isnan(V[0,1]):
                V[0,1] = vr*SP.random.randn(1)
            param = SP.zeros([3])
            L = SP.zeros([2,2])
            #mapping from variance to cholesky factor:
            L[0,0] = SP.sqrt(V[0,0])
            L[1,0] = V[0,1]/L[0,0]                
            L[1,1] = SP.sqrt(V[1,1]-L[0,1]**2)
            if (V[1,1]-L[0,1])>0:
                break
        if 0:
            #check that covariance is correct:
            C = SP.dot(L,L.T)
            assert SP.absolute(C[0,0]-V[0,0])<1E-6, 'outch'
            assert SP.absolute(C[1,1]-V[1,1])<1E-6, 'outch'
            assert SP.absolute(C[0,1]-V[0,1])<1E-6, 'outch'
            assert SP.absolute(C[1,0]-V[0,1])<1E-6, 'outch'       
        #parameters in that order, positive elements on diagonal on log space
        param[0] = SP.log(L[0,0])
        param[1] = L[1,0]
        param[2] = SP.log(L[1,1])
        return param
    
    def _params2var(self,params):
        #1. create cholesky factor
        L = SP.zeros([2,2])
        L[0,0] = SP.exp(params[0])
        L[1,0] = params[1]
        L[1,1] = SP.exp(params[2])       
        C = SP.dot(L,L.T)
        return C
    
                
    def _getParams0(self,**kw_args):
        """get best guess starting points of parameters for optimization"""
        #1. full paramter vector 2 * 4 [Scale, L00, L01, L11]
        cp = SP.zeros([2*4])
        pG = self._var2params(self.VinitG,**kw_args)
        pE = self._var2params(self.VinitE,**kw_args)
        cp[1:4] = pG
        cp[5:8] = pE
        return cp
    
    def _getVarComponents(self,covar_params):
        Cg = self._params2var(covar_params[1:4])
        Cg *= SP.exp(2*covar_params[0])
        Ce = self._params2var(covar_params[5:8])
        Ce *= SP.exp(2*covar_params[4])
        return [Cg,Ce]
        
        
    def fitVariance(self):
        """fit variance components"""
        #1. init GP
        self._init_GP()
        #2. create covar params
        #set start parameters
        for i in xrange(1):
            params = mmtk.CGPHyperParams()
            params['covar'] = self._getParams0(vr=1E-1)
            if i==0:
                self.GP['gp'].setParams(params)            
            self.GP['gpopt'].addOptStartParams(params)
        self.GP['gpopt'].opt()
        
        
        #get optimized hyperparmas
        self.GP['hyperparamsO'] = self.GP['gp'].getParams()
    
        #parse hyperparams into variance components
        [VG,VE] = self._getVarComponents(self.GP['hyperparamsO']['covar'])
        self.VE = VE
        self.VG = VG
        #store covariance for testing
        self.Ktesting = scale_k(self.GP['covar'].K())
        pass
        
    def GWAmain(self,useK=['multi_trait']):
        """main effect GWA, shared accross traits"""
        self.lmm =  mmtk.CLMM()
        if useK=='multi_trait':
            self.lmm.setK(self.Ktesting)
        else:
            self.lmm.setK(self.Kpop)
            
        self.lmm.setSNPs(self.X)
        self.lmm.setPheno(self.Y)
        #covariates: column of ones
        self.lmm.setCovs(self.C)
        #EmmaX mode with useful default settings
        self.lmm.setVarcompApprox0()
        self.lmm.process()
        pv = self.lmm.getPv().flatten()
        return pv
    
    def GWAinter(self,useK=['multi_trait'],I=None,I0=None):
        self.lmi = mmtk.CInteractLMM()
        if useK=='multi_trait':
            self.lmi.setK(self.Ktesting)
        else:
            self.lmi.setK(self.Kpop)
        self.lmi.setSNPs(self.X)
        self.lmi.setPheno(self.Y)
        self.lmi.setCovs(self.C)
        self.lmi.setVarcompApprox0()
        self.lmi.setInter(I)
        self.lmi.setInter0(I0)
        self.lmi.process()
        pvI = self.lmi.getPv().flatten()
        return pvI

        
