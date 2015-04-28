import sys
from limix.utils.utils import smartSum
from limix.core.mean import mean

import pdb
import numpy as NP
import scipy as SP
import scipy.linalg as LA
import time as TIME
from gp_base import gp as GP

class gp2kronSum(GP):

    def __init__(self,mean,Cg,Cn,XX=None,S_XX=None,U_XX=None,offset=1e-4):
        """
        Y:      Phenotype matrix
        Cg:     LIMIX trait-to-trait covariance for genetic contribution
        Cn:     LIMIX trait-to-trait covariance for noise
        XX:     Matrix for fixed sample-to-sample covariance function
        """
        #cache init
        self.cache = {}
        # pheno
        self.setMean(mean)
        # colCovariances
        self.setColCovars(Cg,Cn)
        # row covars
        self.set_XX(XX,S_XX,U_XX)
        #offset for trait covariance matrices
        self.setOffset(offset)
        self.params = None
        self.reml = True
        # time
        self.time = {}
        self.count = {}

    def set_reml(self,value=True):
        self.reml = value

    def get_time(self):
        """ returns time dictionary """
        return self.time

    def get_count(self):
        """ return count dictionary """
        return self.count

    def restart(self):
        """ set all times to 0 """
        for key in self.time.keys():
            self.time[key]  = 0
            self.count[key] = 0

    def setColCovars(self,Cg,Cn):
        """
        set column covariances
        """
        # col covars
        self.Cg = Cg
        self.Cn = Cn

    def setMean(self,mean):
        """
        set gp mean
        """
        self.N, self.P = mean.getDimensions()
        self.mean = mean
        if self.cache.has_key("d"):
            self.mean.d = self.cache['d']
        if self.cache.has_key("Lr"):
            self.mean.Lr = self.cache['Lr']
        if self.cache.has_key("Lc"):
            self.mean.Lc = self.cache['Lc']

    def setY(self,Y):
        """
        set gp mean
        """
        self.mean.Y = Y

    def setOffset(self,offset):
        """
        set offset
        """
        self.offset = offset

    def set_XX(self,XX=None,S_XX=None,U_XX=None):
        """
        set pop struct row covariance
        """
        XXnotNone = XX is not None
        SUnotNone = S_XX is not None and U_XX is not None
        assert XXnotNone or SUnotNone, 'Specify either XX or S_XX and U_XX!'
        if SUnotNone:
            self.cache['Srstar'] = S_XX
            self.cache['Lr'] = U_XX.T
            raise NotImplementedError("self.mean.setRowRotation(Lr=self.cache['Lr'])#this function does not exist")
            self.mean.setRowRotation(Lr=self.cache['Lr'])#this function does not exist
            self.XX_has_changed = False
        else:
            self.XX = XX
            self.XX_has_changed = True

    def getParams(self):
        """
        get hper parameters
        """
        params = {}
        params['Cg'] = self.Cg.getParams()
        params['Cn'] = self.Cn.getParams()
        return params

    def setParams(self,params):
        """
        set hper parameters
        """
        self.params = params
        self.updateParams()

    def updateParams(self):
        """
        update parameters
        """
        keys =self.params.keys()
        if 'Cg' in keys:
            self.Cg.setParams(self.params['Cg'])
        if 'Cn' in keys:
            self.Cn.setParams(self.params['Cn'])

    def _update_cache(self):
        """
        Update cache
        """
        cov_params_have_changed = self.Cg.params_have_changed or self.Cn.params_have_changed

        if self.XX_has_changed:
            start = TIME.time()
            """ Row SVD Bg + Noise """
            self.cache['Srstar'],Urstar  = LA.eigh(self.XX)
            self.cache['Lr']   = Urstar.T
            self.mean.Lr = self.cache['Lr']

            smartSum(self.time,'cache_XXchanged',TIME.time()-start)
            smartSum(self.count,'cache_XXchanged',1)

        if cov_params_have_changed:
            start = TIME.time()
            """ Col SVD Bg + Noise """
            S2,U2 = LA.eigh(self.Cn.K()+self.offset*SP.eye(self.P))
            self.cache['Sc2'] = S2
            US2   = SP.dot(U2,SP.diag(SP.sqrt(S2)))
            USi2  = SP.dot(U2,SP.diag(SP.sqrt(1./S2)))
            Cstar = SP.dot(USi2.T,SP.dot(self.Cg.K(),USi2))
            self.cache['Scstar'],Ucstar = LA.eigh(Cstar)
            self.cache['Lc'] = SP.dot(Ucstar.T,USi2.T)
            """ pheno """
            self.mean.Lc = self.cache['Lc']


        if cov_params_have_changed or self.XX_has_changed:
            """ S """
            self.cache['s'] = SP.kron(self.cache['Scstar'],self.cache['Srstar'])+1
            self.cache['d'] = 1./self.cache['s']
            self.cache['D'] = SP.reshape(self.cache['d'],(self.N,self.P), order='F')

            """ pheno """
            self.mean.d = self.cache['d']
            #self.cache['LZ']  = self.mean.Zstar()
            #self.cache['DLZ'] = self.cache['D']*self.cache['LZ']

            smartSum(self.time,'cache_colSVDpRot',TIME.time()-start)
            smartSum(self.count,'cache_colSVDpRot',1)

        self.XX_has_changed = False
        self.Cg.params_have_changed = False
        self.Cn.params_have_changed = False

    def LML(self, params=None, *kw_args):
        """
        calculate LML
        """
        if params!=None:
            self.setParams(params)

        self._update_cache()

        start = TIME.time()

        #1. const term
        lml  = self.N*self.P*SP.log(2.0*SP.pi)

        #2. logdet term
        lml += SP.sum(SP.log(self.cache['Sc2']))*self.N + SP.log(self.cache['s']).sum()

        #3. quadratic term
        lml += self.mean.var_total() - self.mean.var_explained()[0]

        if self.reml and self.mean.n_fixed_effs>0:
            #4. reml term
            lml += 2*SP.log(SP.diag(self.mean.Areml_chol())).sum()

        lml *= 0.5

        smartSum(self.time,'lml',TIME.time()-start)
        smartSum(self.count,'lml',1)

        return lml




    def LMLgrad(self,params=None,**kw_args):
        """
        LML gradient
        """
        if params is not None:
            self.setParams(params)
        self._update_cache()
        RV = {}
        covars = ['Cg','Cn']
        for covar in covars:
            RV[covar] = self._LMLgrad_covar(covar)
        return RV

    def _LMLgrad_covar(self,covar,**kw_args):
        """
        calculates LMLgrad for covariance parameters
        """
        # precompute some stuff
        if covar=='Cg':
            LRLdiag = self.cache['Srstar']
            n_params = self.Cg.getNumberParams()
        elif covar=='Cn':
            LRLdiag = SP.ones(self.N)
            n_params = self.Cn.getNumberParams()

        # some stuff to cache
        LRLdiag_DLZ = LRLdiag[:,SP.newaxis]*self.mean.DLZ()
        self.mean.LRLdiag = LRLdiag

        # fill gradient vector
        RV = SP.zeros(n_params)
        for i in range(n_params):

            #0. calc LCL
            start = TIME.time()
            if covar=='Cg':     C = self.Cg.Kgrad_param(i)
            elif covar=='Cn':   C = self.Cn.Kgrad_param(i)
            LCL = SP.dot(self.cache['Lc'],SP.dot(C,self.cache['Lc'].T))
            self.mean.LCL = LCL

            #1. der of log det
            start = TIME.time()
            kronDiag  = SP.kron(LCL.diagonal(),LRLdiag)
            RV[i] = SP.dot(self.cache['d'],kronDiag)
            smartSum(self.time,'lmlgrad_trace',TIME.time()-start)
            smartSum(self.count,'lmlgrad_trace',1)

            #2. der of quad form
            start = TIME.time()
            KDLZ  = SP.dot(LRLdiag_DLZ,LCL.T)
            KDLZ += self.mean.Xstar_beta_grad()
            RV[i] -= (self.mean.DLZ()*KDLZ).sum()


            smartSum(self.time,'lmlgrad_quadform',TIME.time()-start)
            smartSum(self.count,'lmlgrad_quadform',1)

            if self.reml and self.mean.n_fixed_effs>0:
                # der of log det reml
                RV[i] += SP.einsum('ij,ji->',self.mean.Areml_inv(),self.mean.Areml_grad())

            RV[i] *= 0.5

        return RV

    def LMLgrad_debug(self,**kw_args):
        """
        LML gradient debug
        """
        RV = {}
        covars = ['Cg','Cn']
        for covar in covars:
            RV[covar] = self._LMLgrad_covar_debug(covar)
        return RV

    def _LMLgrad_covar_debug(self,covar):

        assert self.N*self.P<2000, 'gp2kronSum:: N*P>=2000'

        y  = SP.reshape(self.Y,(self.N*self.P), order='F')

        K  = SP.kron(self.Cg.K(),self.XX)
        K += SP.kron(self.Cn.K()+self.offset*SP.eye(self.P),SP.eye(self.N))

        cholK = LA.cholesky(K).T
        Ki  = LA.cho_solve((cholK,True),SP.eye(y.shape[0]))
        Kiy  = LA.cho_solve((cholK,True),y)

        if covar=='Cr':     n_params = self.Cr.getNumberParams()
        elif covar=='Cg':   n_params = self.Cg.getNumberParams()
        elif covar=='Cn':   n_params = self.Cn.getNumberParams()

        RV = SP.zeros(n_params)

        for i in range(n_params):
            #0. calc grad_i
            if covar=='Cg':
                C   = self.Cg.Kgrad_param(i)
                Kgrad  = SP.kron(C,self.XX)
            elif covar=='Cn':
                C   = self.Cn.Kgrad_param(i)
                Kgrad  = SP.kron(C,SP.eye(self.N))

            #1. der of log det
            RV[i]  = 0.5*(Ki*Kgrad).sum()

            #2. der of quad form
            RV[i] -= 0.5*(Kiy*SP.dot(Kgrad,Kiy)).sum()

        return RV

    def predict(self,XXstar):
        """
        Make predictions:
            XXstar:     cross covariance matrix Nstar,N
        """
        self._update_cache()
        KiY = SP.dot(self.cache['Lr'].T,SP.dot(self.cache['DLY'],self.cache['Lc']))
        rv = SP.dot(XXstar,SP.dot(KiY,self.Cg.K()))
        return rv


    """ debug functions """

    def check_Areml(self):
        self.LML()
        K  = SP.kron(self.Cg.K(),self.XX)
        K += SP.kron(self.Cn.K()+self.offset*SP.eye(self.P),SP.eye(self.N))
        cholK = LA.cholesky(K).T
        Ki    = LA.cho_solve((cholK,True),SP.eye(self.N*self.P))
        X = []
        for term_i in range(self.mean.n_terms):
            X.append(SP.kron(self.mean.A[term_i].T,self.mean.F[term_i]))
        X = SP.concatenate(X,1)
        Areml = SP.dot(X.T,SP.dot(Ki,X))
        print ((Areml-self.mean.Areml())**2).mean()<1e-6

    def check_beta_hat(self):
        self.LML()
        y  = self.mean.Y.reshape((self.mean.Y.size,1),order='F')
        K  = SP.kron(self.Cg.K(),self.XX)
        K += SP.kron(self.Cn.K()+self.offset*SP.eye(self.P),SP.eye(self.N))
        cholK = LA.cholesky(K).T
        Ki    = LA.cho_solve((cholK,True),SP.eye(self.N*self.P))
        X = []
        for term_i in range(self.mean.n_terms):
            X.append(SP.kron(self.mean.A[term_i].T,self.mean.F[term_i]))
        X = SP.concatenate(X,1)
        XKiy = SP.dot(X.T,SP.dot(Ki,y))
        beta_hat = SP.dot(self.mean.Areml_inv(),XKiy)
        L = SP.kron(self.cache['Lc'],self.cache['Lr'])
        zstar = SP.dot(L,self.mean.Y.reshape((self.mean.Y.size,1),order='F')-SP.dot(X,beta_hat))
        zstar1 = self.mean.Zstar().reshape((self.mean.Y.size,1),order='F')

        # beta 1
        #L = SP.kron(self.cache['Lc'],self.cache['Lr'])
        #Xstar = SP.dot(L,X)
        #yhat  = SP.dot(L,self.mean.Y.reshape((self.mean.Y.size,1)))
        #yhat *= self.cache['d'][:,SP.newaxis]
        #beta_hat1 = SP.dot(self.mean.Areml_inv(),SP.dot(Xstar.T,yhat))

        print ((beta_hat-self.mean.beta_hat())**2).mean()<1e-6
        print ((zstar-zstar1)**2).mean()<1e-6

    def LMLdebug(self):
        """
        LML function for debug
        """
        assert self.N*self.P<2000, 'gp2kronSum:: N*P>=2000'

        y  = SP.reshape(self.mean.Y,(self.N*self.P,1), order='F')

        K  = SP.kron(self.Cg.K(),self.XX)
        K += SP.kron(self.Cn.K()+self.offset*SP.eye(self.P),SP.eye(self.N))

        cholK = LA.cholesky(K)

        X = []
        for term_i in range(self.mean.n_terms):
            X.append(SP.kron(self.mean.A[term_i].T,self.mean.F[term_i]))
        X = SP.concatenate(X,1)
        z = y-SP.dot(X,self.mean.beta_hat())

        Kiz   = LA.cho_solve((cholK,False),z)

        lml  = y.shape[0]*SP.log(2*SP.pi)
        lml += 2*SP.log(SP.diag(cholK)).sum()
        lml += (z*Kiz).sum()
        lml += 2*SP.log(SP.diag(self.mean.Areml_chol())).sum()
        lml *= 0.5

        return lml

    def check_Agrad(self):
        """
        A = X.T Ki X
        Agrad = - X.T Ki dK Ki X
        """
        self.LML()
        K  = SP.kron(self.Cg.K(),self.XX)
        K += SP.kron(self.Cn.K()+self.offset*SP.eye(self.P),SP.eye(self.N))
        cholK = LA.cholesky(K).T
        Ki    = LA.cho_solve((cholK,True),SP.eye(self.N*self.P))
        X = []
        for term_i in range(self.mean.n_terms):
            X.append(SP.kron(self.mean.A[term_i].T,self.mean.F[term_i]))
        X = SP.concatenate(X,1)

        i = 0
        C   = self.Cg.Kgrad_param(i)
        Kgrad = SP.kron(C,self.XX)

        Agrad = -SP.dot(X.T,SP.dot(Ki,SP.dot(Kgrad,SP.dot(Ki,X))))

        # Agrad1 = -Xstar.T D LCL\kronLRLdiag Xhat
        LRLdiag = self.cache['Srstar']
        LCL = SP.dot(self.cache['Lc'],SP.dot(C,self.cache['Lc'].T))
        #dK    = SP.kron(LCL,SP.diag(LRLdiag))
        #Xstar = self.mean.Xstar()
        #Xhat  = self.mean.Xhat()
        #DKiXhat = self.cache['d'][:,SP.newaxis]*SP.dot(dK,Xhat)
        #Agrad2 = -SP.dot(Xstar.T,DKiXhat)

        self.mean.LRLdiag = LRLdiag
        self.mean.LCL = LCL
        Agrad1 = self.mean.Areml_grad()

        print ((Agrad-Agrad1)**2).mean()<1e-6

    def check_beta_grad(self):
        """
        b = Ai Xt Ki y
        bgrad = - Ai dA beta
                - Ai Xt Ki dK Ki y
        """
        self.LML()
        y  = SP.reshape(self.mean.Y,(self.N*self.P,1), order='F')
        K  = SP.kron(self.Cg.K(),self.XX)
        K += SP.kron(self.Cn.K()+self.offset*SP.eye(self.P),SP.eye(self.N))
        cholK = LA.cholesky(K).T
        Ki    = LA.cho_solve((cholK,True),SP.eye(self.N*self.P))
        X = []
        for term_i in range(self.mean.n_terms):
            X.append(SP.kron(self.mean.A[term_i].T,self.mean.F[term_i]))
        X = SP.concatenate(X,1)

        i = 0
        C   = self.Cg.Kgrad_param(i)
        Kgrad = SP.kron(C,self.XX)

        Agrad = -SP.dot(X.T,SP.dot(Ki,SP.dot(Kgrad,SP.dot(Ki,X))))

        beta_grad = SP.dot(Agrad,self.mean.beta_hat())
        beta_grad+= SP.dot(X.T,SP.dot(Ki,SP.dot(Kgrad,SP.dot(Ki,y))))
        beta_grad = -SP.dot(self.mean.Areml_inv(),beta_grad)
        Xstar_beta_grad = SP.dot(self.mean.Xstar(),beta_grad)

        LRLdiag = self.cache['Srstar']
        LCL = SP.dot(self.cache['Lc'],SP.dot(C,self.cache['Lc'].T))
        self.mean.LRLdiag = LRLdiag
        self.mean.LCL = LCL
        beta_grad1 = self.mean.beta_grad()
        Xstar_beta_grad1 = self.mean.Xstar_beta_grad().reshape((self.P*self.N,1),order='F')

        print ((beta_grad-beta_grad1)**2).mean()<1e-6
        print ((Xstar_beta_grad-Xstar_beta_grad1)**2).mean()<1e-6
