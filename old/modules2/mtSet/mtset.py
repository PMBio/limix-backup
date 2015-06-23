import sys
sys.path.append('./../../..')
from limix.utils.preprocess import remove_dependent_cols
from limix.utils.util_functions import smartDumpDictHdf5

# core 
from limix.core.gp import GP2KronSum
from limix.core.gp import GP2KronSumLR
from limix.core.gp import GP3KronSumLR
from limix.core.covar import FreeFormCov 

import h5py
import pdb
import scipy as sp
import scipy.stats as st
import time as TIME
import copy
import warnings
import os


class MTSet():
    """
    This class eases implementation of efficient set test algorithms.
    mtSet can account for relatedness and can be used for single- and multi-trait analysis
    """

    def __init__(self, Y=None, R=None, S_R=None, U_R=None, traitID=None, F=None, rank = 1):
        """
        Args:
            Y:          [N, P] phenotype matrix
            F:          [N, K] matrix of fixed effect design.
                        K is the number of per-trait covariates.
            R:          [N, N] genetic relatedness matrix between individuals.
                        In alternative to R, S_R and U_R can be provided.
                        If not specified a model without relatedness component is considered.
            S_R:        N vector of eigenvalues of R
            U_R:        [N, N] eigenvector matrix of R
            traiID:     P vector of the IDs of the phenotypes to analyze (optional)
            rank:       rank of the trait covariance matrix of the variance component to be tested (default is 1)
        """
        # data
        noneNone = S_R is not None and U_R is not None
        self.bgRE = R is not None or noneNone
        # fixed effect
        msg = 'The current implementation of the full rank mtSet'
        msg+= ' does not support covariates.'
        msg+= ' We reccommend to regress out covariates and'
        msg+= ' subsequently quantile normalize the phenotypes'
        msg+= ' to a normal distribution prior to use mtSet.'
        msg+= ' This can be done within the LIMIX framework using'
        msg+= ' the methods limix.utils.preprocess.regressOut and'
        msg+= ' limix.utils.preprocess.gaussianize'
        assert not (F is not None and self.bgRE), msg
        if F is not None:
            F = remove_dependent_cols(F)
        #traitID
        if traitID is None:
            traitID = sp.array(['trait %d' % p for p in range(Y.shape[1])])
        self.setTraitID(traitID)
        #init covariance matrices and gp
        Cg = FreeFormCov(Y.shape[1])
        Cn = FreeFormCov(Y.shape[1])
        G  = 1. * (sp.rand(Y.shape[0],1)<0.2)
        self._gp = GP3KronSumLR(Y=Y, Cg=Cg, Cn=Cn, S_R=S_R, U_R=U_R, G=G, rank = 1)
        # null model params
        self.null = None
        # calls itself for column-by-column trait analysis
        self.mtssST = None
        self.nullST = None
        self.infoOpt   = None
        self.infoOptST = None
        pass

    ##################################################
    # Properties
    ##################################################
    @property
    def N(self):    return self._gp.covar.dim_r

    @property
    def P(self):    return self._gp.covar.dim_c

    @property
    def Cr(self):   return self._gp.covar.Cr

    @property
    def Cg(self):   return self._gp.covar.Cg

    @property
    def Cn(self):   return self._gp.covar.Cn

    @property
    def Y(self):    return self._gp.mean.Y

    @property
    def S_R(self):
        if self.bgRE:   RV = self._gp.covar.Sr()
        else:           RV = None
        return RV

    @property
    def U_R(self):
        if self.bgRE:   RV = self._gp.covar.Lr().T
        else:           RV = None
        return RV

    #########################################
    # Setters
    #########################################
    @Y.setter
    def Y(self, value):
        assert value.shape[0]==self.N, 'Dimension mismatch'
        assert value.shape[1]==self.P, 'Dimension mismatch'
        self._gp.mean.Y = Y

    def setTraitID(self,traitID):
        self.traitID = traitID

    ################################################
    # Fitting null model
    ###############################################
    def fitNull(self,verbose=True,cache=False,out_dir='./cache',fname=None,rewrite=False,seed=None,n_times=10,factr=1e3,init_method=None):
        """
        Fit null model
        """
        if seed is not None:    sp.random.seed(seed)

        pdb.set_trace()

        read_from_file = False
        if cache:
            assert fname is not None, 'MultiTraitSetTest:: specify fname'
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            out_file = os.path.join(out_dir,fname)
            read_from_file = os.path.exists(out_file) and not rewrite

        RV = {}
        if read_from_file:
            f = h5py.File(out_file,'r')
            for key in f.keys():
                RV[key] = f[key][:]
            f.close()
            self.setNull(RV)
        else:
            start = TIME.time()
            if self.bgRE:
                self._gpNull = GP2KronSum(Y=self.Y, F=None, A=None, Cg=self.Cg, Cn=self.Cn, R=None, S_R=self.S_R, U_R=self.U_R)
                self._gpNull.covar.setRandomParams()
                self._gpNull.optimize()
            else:
                self.gpNull = gp2kronSumLR(self.Y,self.Cn,Xr=sp.ones((self.N,1)),F=self.F)
            for i in range(n_times):
                params0,Ifilter=self._initParams(init_method=init_method)
                conv,info = OPT.opt_hyper(self.gpNull,params0,Ifilter=Ifilter,factr=factr)
                if conv: break
            if not conv:    warnings.warn("not converged")
            LMLgrad = sp.concatenate([self.gpNull.LMLgrad()[key]**2 for key in self.gpNull.LMLgrad().keys()]).mean()
            LML = self.gpNull.LML()
            if 'mean' in params0.keys():
                RV['params_mean'] = self.gpNull.mean.getParams()
            RV['params0_g'] = self.Cg.getParams()
            RV['params0_n'] = self.Cn.getParams()
            RV['Cg'] = self.Cg.K()
            RV['Cn'] = self.Cn.K()
            RV['conv'] = sp.array([conv])
            RV['time'] = sp.array([TIME.time()-start])
            RV['NLL0'] = sp.array([LML])
            RV['LMLgrad'] = sp.array([LMLgrad])
            RV['nit'] = sp.array([info['nit']])
            RV['funcalls'] = sp.array([info['funcalls']])
            if self.bgRE:
                RV['h2'] = self.gpNull.h2()
                RV['h2_ste'] = self.gpNull.h2_ste()
                RV['Cg_ste'] = self.gpNull.ste('Cg')
                RV['Cn_ste'] = self.gpNull.ste('Cn')
            self.null = RV
            if cache:
                f = h5py.File(out_file,'w')
                dumpDictHdf5(RV,f)
                f.close()
        return RV

    def getNull(self):
        """ get null model info """
        return self.null

    def setNull(self,null):
        """ set null model info """
        self.null = null

    ###########################################
    # Fitting alternative model
    ###########################################

    def optimize(self,Xr,params0=None,n_times=10,verbose=True,vmax=5,perturb=1e-3,factr=1e3):
        """
        Optimize the model considering Xr
        """
        # set params0 from null if params0==Null
        if params0 is None:
            if self.null is None:
                if verbose:     print ".. fitting null model upstream"
                self.fitNull()
            if self.bgRE:
                params0 = {'Cg':self.null['params0_g'],'Cn':self.null['params0_n']}
            else:
                params0 = {'Cn':self.null['params0_n']}
            if 'params_mean' in self.null:
                if self.null['params_mean'].shape[0]>0:
                    params0['mean'] = self.null['params_mean']
            params_was_None = True
            
        else:
            params_was_None = False
        Xr *= sp.sqrt(self.N/(Xr**2).sum())
        self.gp.set_Xr(Xr)
        self.gp.restart()
        start = TIME.time()
        for i in range(n_times):
            if params_was_None:
                n_params = self.Cr.getNumberParams()
                params0['Cr'] = 1e-3*sp.randn(n_params)
            conv,info = OPT.opt_hyper(self.gp,params0,factr=factr)
            conv *= self.gp.Cr.K().diagonal().max()<vmax
            conv *= self.getLMLgrad()<0.1
            if conv or not params_was_None: break
        self.infoOpt = info
        if not conv:
            warnings.warn("not converged")
        # return value
        RV = {}
        if self.P>1:
            RV['Cr']  = self.getCr()
            if self.bgRE: RV['Cg']  = self.getCg()
            RV['Cn']  = self.getCn()
        RV['time']  = sp.array([TIME.time()-start])
        RV['params0'] = params0
        RV['nit'] = sp.array([info['nit']])
        RV['funcalls'] = sp.array([info['funcalls']])
        RV['var']    = self.getVariances()
        RV['conv']  = sp.array([conv])
        RV['NLLAlt']  = sp.array([self.getNLLAlt()])
        RV['LLR']    = sp.array([self.getLLR()])
        RV['LMLgrad'] = sp.array([self.getLMLgrad()])
        return RV

    def getInfoOpt(self):
        """ get information for the optimization """
        return self.infoOpt

    def getTimeProfiling(self):
        """ get time profiling """
        rv = {'time':self.gp.get_time(),'count':self.gp.get_count()}
        return rv

    def getCr(self):
        """
        get estimated region trait covariance
        """
        assert self.P>1, 'this is a multitrait model'
        return self.gp.Cr.K()
        
    def getCg(self):
        """
        get estimated genetic trait covariance
        """
        assert self.P>1, 'this is a multitrait model'
        return self.gp.Cg.K()

    def getCn(self):
        """
        get estimated noise trait covariance
        """
        assert self.P>1, 'this is a multitrait model'
        return self.gp.Cn.K()

    def getVariances(self):
        """
        get variances
        """
        if self.P==1:
            params = self.gp.getParams()
            if self.bgRE:       keys = ['Cr','Cg','Cn']
            else:               keys = ['Cr','Cn']
            var = sp.array([params[key][0]**2 for key in keys])
        else:
            var = []
            var.append(self.getCr().diagonal())
            if self.bgRE:
                var.append(self.getCg().diagonal())
            var.append(self.getCn().diagonal())
            var = sp.array(var)
        return var

    def getNLLAlt(self):
        """
        get negative log likelihood of the alternative
        """
        return self.gp.LML()

    def getLLR(self):
        """
        get log likelihood ratio
        """
        assert self.null is not None, 'null model needs to be fitted!'
        return self.null['NLL0'][0]-self.getNLLAlt()

    def getLMLgrad(self):
        """
        get norm LML gradient
        """
        LMLgrad = self.gp.LMLgrad()
        lmlgrad  = 0
        n_params = 0
        for key in LMLgrad.keys():
            lmlgrad  += (LMLgrad[key]**2).sum()
            n_params += LMLgrad[key].shape[0]
        lmlgrad /= float(n_params)
        return lmlgrad

    def fitNullTraitByTrait(self,verbose=True,cache=False,out_dir='./cache',fname=None,rewrite=False):
        """
        Fit null model trait by trait
        """
        read_from_file = False
        if cache:
            assert fname is not None, 'MultiTraitSetTest:: specify fname'
            if not os.path.exists(out_dir): os.makedirs(out_dir)
            out_file = os.path.join(out_dir,fname)
            read_from_file = os.path.exists(out_file) and not rewrite

        RV = {}
        if read_from_file:
            f = h5py.File(out_file,'r')
            for p in range(self.P):
                trait_id = self.traitID[p]
                g = f[trait_id]
                RV[trait_id] = {}
                for key in g.keys():
                    RV[trait_id][key] = g[key][:]
            f.close()
            self.nullST=RV
        else:
            """ create mtssST and fit null column by column returns all info """
            if self.mtssST is None:
                y = sp.zeros((self.N,1)) 
                self.mtssST = MultiTraitSetTest(y,R=self.R,S_R=self.S_R,U_R=self.U_R,F=self.F)
            RV = {}
            for p in range(self.P):
                trait_id = self.traitID[p]
                y = self.Y[:,p:p+1]
                self.mtssST._setY(y)
                RV[trait_id] = self.mtssST.fitNull()
            self.nullST = RV
            if cache:
                f = h5py.File(out_file,'w')
                smartDumpDictHdf5(RV,f)
                f.close()
        return RV

    def optimizeTraitByTrait(self,Xr,verbose=True,n_times=10,factr=1e3):
        """ Optimize trait by trait """
        assert self.nullST is not None, 'fit null model beforehand'
        if self.mtssST is None:
            y = sp.zeros((self.N,1)) 
            self.mtssST = MultiTraitSetTest(y,R=self.R,S_R=self.S_R,U_R=self.U_R,F=self.F)
        RV = {}
        self.infoOptST = {}
        self.timeProfilingST = {}
        for p in range(self.P):
            y = self.Y[:,p:p+1]
            trait_id = self.traitID[p]
            self.mtssST._setY(y)
            self.mtssST.setNull(self.nullST[trait_id])
            RV[trait_id] = self.mtssST.optimize(Xr,n_times=n_times,factr=factr)
            self.infoOptST[trait_id] = self.mtssST.getInfoOpt()
            self.timeProfilingST[trait_id] = self.mtssST.getTimeProfiling()
        return RV

    def getInfoOptST(self):
        """ get information for the optimization """
        return self.infoOptST

    def getTimeProfilingST(self):
        """ get time profiling """
        return self.timeProfilingST

    def _initParams(self,init_method=None):
        """ this function initializes the paramenter and Ifilter """
        if self.P==1:
            if self.bgRE:
                params0 = {'Cg':sp.sqrt(0.5)*sp.ones(1),'Cn':sp.sqrt(0.5)*sp.ones(1)}
                Ifilter = None
            else:
                params0 = {'Cr':1e-9*sp.ones(1),'Cn':sp.ones(1)}
                Ifilter = {'Cr':sp.zeros(1,dtype=bool),'Cn':sp.ones(1,dtype=bool)}
        else:
            if self.bgRE:
                if self.colCovarType_g=='freeform' and self.colCovarType_n=='freeform':
                    if init_method=='pairwise':
                        _RV = fitPairwiseModel(self.Y,R=self.R,S_R=self.S_R,U_R=self.U_R,verbose=False)
                        params0 = {'Cg':_RV['params0_Cg'],'Cn':_RV['params0_Cn']}
                    elif init_method=='random':
                        params0 = {'Cg':sp.randn(self.Cg.getNumberParams()),'Cn':sp.randn(self.Cn.getNumberParams())}
                    else:
                        cov = 0.5*sp.cov(self.Y.T)+1e-4*sp.eye(self.P)
                        chol = LA.cholesky(cov,lower=True)
                        params = chol[sp.tril_indices(self.P)]
                        params0 = {'Cg':params.copy(),'Cn':params.copy()}
            
                Ifilter = None
                
            else:
                if self.colCovarType_n=='freeform':
                    cov = sp.cov(self.Y.T)+1e-4*sp.eye(self.P)
                    chol = LA.cholesky(cov,lower=True)
                    params = chol[sp.tril_indices(self.P)]
                #else:
                #    S,U=LA.eigh(cov)
                #    a = sp.sqrt(S[-self.rank_r:])[:,sp.newaxis]*U[:,-self.rank_r:]
                #    if self.colCovarType=='lowrank_id':
                #        c = sp.sqrt(S[:-self.rank_r].mean())*sp.ones(1)
                #    else:
                #        c = sp.sqrt(S[:-self.rank_r].mean())*sp.ones(self.P)
                #    params0_Cn = sp.concatenate([a.T.ravel(),c])
                params0 = {'Cr':1e-9*sp.ones(self.P),'Cn':params}
                Ifilter = {'Cr':sp.zeros(self.P,dtype=bool),
                            'Cn':sp.ones(params.shape[0],dtype=bool)}
        if self.mean.F is not None and self.bgRE:
            params0['mean'] = 1e-6*sp.randn(self.mean.getParams().shape[0])
            if Ifilter is not None:
                Ifilter['mean'] = sp.ones(self.mean.getParams().shape[0],dtype=bool)
        return params0,Ifilter

if __name__=='__main__':
    from limix.utils.preprocess import covar_rescale 

    sp.random.seed(0)

    #generate phenotype
    n = 1000
    p = 4
    f = 10
    Y = sp.randn(n, p)
    X = sp.randn(n, f)
    G = sp.randn(n, f)
    R = sp.dot(X, X.T)
    R = covar_rescale(R)
    
    pdb.set_trace()

    mts = mtset(Y, R=R)
    nullMTInfo = mts.fitNull(cache=False)
    mts.optimize(G)

