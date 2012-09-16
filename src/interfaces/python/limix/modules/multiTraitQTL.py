"""testing script for multi trait covariance stuff"""
import sys
sys.path.append('./..')
sys.path.append('./../../..')

import scipy as SP
import scipy.linalg
import limix
import time
import pdb
import pylab as PL
import utils.preprocess as preprocess





class CMMT(object):
    def __init__(self, X=None, E=None, Y=None, C=None, Ks=None, Kgeno=None, T=2, standardize=True, Ve00=SP.nan, Ve11=SP.nan, Ve01=SP.nan, Vg00=SP.nan, Vg01=SP.nan, Vg11=SP.nan):
        """
        X: SNPS [N x S]
        Y: phenotype [N x 1]
        E: env. indocator [N x 1]
        C: covariates [N x C]
        T: number of traits (2)
        Ks[XX^T]: covariances to fit [N x N]
        Kgeno: genotype identity covariance [N x N]
        standardize[True]: standardize phenotype per trait?
        """
        assert T == 2, 'supporting 2 env. only now'
        assert Y.shape[1] == 1, 'Y needs to be [N x 1]'
        if X != None: #Allow users only to estimate variance components
            assert X.shape[0] == Y.shape[0], 'X and Y have incompatible shape'
        assert E.shape[0] == Y.shape[0], 'E and Y have incompatible shape'
        if Ks is None:
            Ks = [1.0 / X.shpape[1] * SP.dot(X, X.T)]
        for K in Ks:
            assert K.shape[0] == Y.shape[0], 'Kpop and Y have incompatible shape'
        assert Kgeno.shape[0] == Y.shape[0], 'Kgeno and Y have incompatible shape'
        assert len(SP.unique(E)) == 2, 'Only supporting 2 environments (0/1) right now'
        assert 0 in E, 'Only supporting 2 environments (0/1) right now'
        assert 1 in E, 'Only supporting 2 environments (0/1) right now'

        if C is None:
            C = SP.ones([Y.shape[0], 1])

        self.X = X
        self.E = E
        self.Y = Y
        self.T = T
        self.C = C
        self.Ks    = [preprocess.scale_K(K) for K in Ks]
        self.Kgeno = preprocess.scale_K(Kgeno)

        #variance component init:
        self.VinitE = SP.array([[Ve00, Ve01], [Ve01, Ve11]])
        self.VinitG = [SP.array([[Vg00, Vg01], [Vg01, Vg11]])]*len(self.Ks)

        #standardize data
        if standardize:
            self._standardize()

        #init lmm object: testing
        self.lmm =  limix.CLMM()
        self.lmi = limix.CInteractLMM()
        
        #fit variance components using delta - fitting
        if SP.array([SP.isnan(V).any() for V in self.VinitG]).any():
            self._init_variance()


    def _standardize(self):
        """standardize expression per environment"""
        for t in xrange(self.T):
            Ie = (self.E[:, 0] == t)
            _Y = self.Y[Ie]
            _Y -= _Y.mean()
            _Y /= _Y.std()
            self.Y[Ie] = _Y
        pass

    def _init_variance(self):
        """guess variance components based on independent analysis"""
        VsG = []
        VsE = []
        for ik in xrange(len(self.Ks)):
            D = []
            K = self.Ks[ik]
            VG = SP.zeros([2,2])
            VE = SP.zeros([2,2])
            for t in xrange(self.T):
                Ie = (self.E[:, 0] == t)
                _Y = self.Y[Ie]
                _C = self.C[Ie]
                _X = SP.zeros([_C.shape[0], 0])
                self.lmm.setK(K[Ie, :][:, Ie])
                self.lmm.setSNPs(_X)
                self.lmm.setCovs(_C)
                self.lmm.setPheno(_Y)
                #optimize delta -10..10
                self.lmm.setVarcompApprox0(-20, 10, 1000)
                self.lmm.process()
                #get delta
                ldelta = self.lmm.getLdelta0().flatten()
                D.append(SP.exp(ldelta))
                pass
            VG[0, 0] = 1. / (1 + D[0])
            VG[1, 1] = 1. / (1 + D[1])
            VE[0, 0] = D[0] / (1 + D[0])
            VE[1, 1] = D[1] / (1 + D[1])
            VsG.append(VG)
            VsE.append(VE)
        #use all fitted genotype models as staring point
        self.VinitG = VsG
        #arbitrarily use first environment model
        self.VinitE = VsE[0]

    def _init_GP(self):
        """create GP instance for fitting"""
        GP = {}
        #overall covariace
        GP['covar'] = limix.CSumCF()
        
        #1. genotype X env covariances
        GP['covar_GG']= []
        GP['covar_GE']= []
        GP['covar_G']= []
        for K in self.Ks:
            _covar_gg = limix.CFixedCF(K)
            #freeform covariance: requiring number of traits/group (T)
            _covar_ge = limix.CFreeFormCF(self.T)
            _covar_g = limix.CProductCF()
            _covar_g.addCovariance(_covar_gg)
            _covar_g.addCovariance(_covar_ge)
            #add to list
            GP['covar_GG'].append(_covar_gg)
            GP['covar_GE'].append(_covar_ge)
            GP['covar_G'].append(_covar_g)

        #2. env covariance:
        GP['covar_EG'] = limix.CFixedCF(self.Kgeno)
        #freeform covariance: requiring number of traits/group (T)
        GP['covar_EE'] = limix.CFreeFormCF(self.T)
        GP['covar_E'] = limix.CProductCF()
        GP['covar_E'].addCovariance(GP['covar_EG'])
        GP['covar_E'].addCovariance(GP['covar_EE'])

        #add to sum CF
        #1. genotype covariances
        for ig in xrange(len(GP['covar_G'])):
            GP['covar'].addCovariance(GP['covar_G'][ig])
        #2. environment covariance
        GP['covar'].addCovariance(GP['covar_E'])

        #liklihood: NULL likleihhod; all variance is explained by covariance function.
        GP['ll'] = limix.CLikNormalNULL()
        GP['hyperparams'] = limix.CGPHyperParams()
        #Create GP instance
        GP['gp']=limix.CGPbase(GP['covar'],GP['ll'])
        #set data
        GP['gp'].setY(self.Y)
        #create X: we have 2 covariances that need inputs (Cover_GEs,covar_GE)
        Xgp = self.E
        #add term for each genotype covariance
        for ig in xrange(len(GP['covar_G'])):
            Xgp = SP.concatenate((Xgp,self.E),axis=1)
        GP['gp'].setX(Xgp)
        #optimization interface
        GP['gpopt'] = limix.CGPopt(GP['gp'])

        #filter?
        #mask out scaling parameters for all covariance sum components
        covar_mask = SP.ones([GP['covar'].getNumberParams(),1])
        for i in xrange((1+len(self.Ks))):
            covar_mask[i*4] = 0
        mask = limix.CGPHyperParams()
        mask['covar'] = covar_mask
        GP['gpopt'].setParamMask(mask)
        self.GP=GP
    
    
    def _var2params(self,V,vr=1E-4):
        """
        Create param object from variance component array
        Missing values will be replaced by random values
        """
        while 1 == 1:
            if SP.isnan(V[0, 1]):
                V[0, 1] = vr * SP.random.randn(1)
            param = SP.zeros([3])
            L = SP.zeros([2, 2])
            #mapping from variance to cholesky factor:
            L[0, 0] = SP.sqrt(V[0, 0])
            L[1, 0] = V[0, 1] / L[0, 0]
            L[1, 1] = SP.sqrt(V[1, 1] - L[0, 1] ** 2)
            if (V[1, 1] - L[0, 1]) > 0:
                break
        if 0:
            #check that covariance is correct:
            C = SP.dot(L, L.T)
            assert SP.absolute(C[0, 0] - V[0, 0]) < 1E-6, 'outch'
            assert SP.absolute(C[1, 1] - V[1, 1]) < 1E-6, 'outch'
            assert SP.absolute(C[0, 1] - V[0, 1]) < 1E-6, 'outch'
            assert SP.absolute(C[1, 0] - V[0, 1]) < 1E-6, 'outch'
        #parameters in that order, positive elements on diagonal on log space
        param[0] = SP.log(L[0, 0])
        param[1] = L[1, 0]
        param[2] = SP.log(L[1, 1])
        return param

    def _params2var(self, params):
        #1. create cholesky factor
        L = SP.zeros([2, 2])
        L[0, 0] = SP.exp(params[0])
        L[1, 0] = params[1]
        L[1, 1] = SP.exp(params[2])
        C = SP.dot(L, L.T)
        return C


    def _getParams0(self, **kw_args):
        """get best guess starting points of parameters for optimization"""
        #1. full paramter vector 2 * 4 [Scale, L00, L01, L11]
        cp = SP.zeros([(1+len(self.Ks)) * 4])
        for ig in xrange(len(self.Ks)):
            pG = self._var2params(self.VinitG[ig], **kw_args)
            i0 = ig*4+1
            i1 = ig*4+1+3
            cp[i0:i1] = pG
        pE = self._var2params(self.VinitE, **kw_args)
        cp[i1+1:i1+4] = pE
        return cp

    def _getVarComponents(self, covar_params):
        Ve = None
        Vg = []
        for i in xrange(len(self.Ks)):
            i0 = i*4+1
            i1 = i*4+1+3
            Cg = self._params2var(covar_params[i0:i1])
            #scaling (should be = 1):
            Cg *= SP.exp(2 * covar_params[i0-1])
            Vg.append(Cg)
        Ve = self._params2var(covar_params[i1+1:i1+4])
        Ve *= SP.exp(2 * covar_params[i1])
        return [Vg, Ve]


    def fitVariance(self):
        """fit variance components"""
        #1. init GP
        self._init_GP()
        #2. create covar params
        #set start parameters
        for i in xrange(1):
            params0 = limix.CGPHyperParams()
            params0['covar'] = self._getParams0(vr=1E-1)
            if i == 0:
                self.GP['gp'].setParams(params0)
            self.GP['gpopt'].addOptStartParams(params0)
        self.GP['gpopt'].opt()
        #get gradients and check that they are~0
        lmlgrad=self.GP['gp'].LMLgrad()
        assert lmlgrad['covar'].max() < 1E-3, 'optimization not propperly converged'

        #get optimized hyperparmas
        self.GP['hyperparamsO'] = self.GP['gp'].getParams()

        #parse hyperparams into variance components
        [VG, VE] = self._getVarComponents(params0['covar'])
        self.VE0 = VE
        self.VG0 = VG

        #parse hyperparams into variance components
        [VG, VE] = self._getVarComponents(self.GP['hyperparamsO']['covar'])
        self.VE = VE
        self.VG = VG
        #store covariance for testing
        self.Ktesting = preprocess.scale_K(self.GP['covar'].K())
        pass

    def GWAmain(self, useK=['multi_trait']):
        """main effect GWA, shared accross traits"""
        self.lmm =  limix.CLMM()
        if useK=='multi_trait':
            self.lmm.setK(self.Ktesting)
        else:
            self.lmm.setK(self.Ks[0])

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
        self.lmi = limix.CInteractLMM()
        if useK=='multi_trait':
            self.lmi.setK(self.Ktesting)
        else:
            self.lmi.setK(self.Ks[0])
        self.lmi.setSNPs(self.X)
        self.lmi.setPheno(self.Y)
        self.lmi.setCovs(self.C)
        self.lmi.setVarcompApprox0()
        self.lmi.setInter(I)
        self.lmi.setInter0(I0)
        self.lmi.process()
        pvI = self.lmi.getPv().flatten()
        return pvI


