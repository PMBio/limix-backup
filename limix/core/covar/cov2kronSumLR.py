import sys
from covar_base import Covariance
from limix.core.covar import LowRankCov
from limix.core.type.cached import cached
import scipy as sp
import scipy.linalg as la
import numpy.linalg as nla
import warnings

import pdb

class Cov2KronSumLR(Covariance):

    def __init__(self, Cn = None, G = None, rank = 1):
        self.setColCovars(Cn, rank = rank)
        self.G = G
        self.dim = self.dim_c * self.dim_r
        self._calcNumberParams()
        self._use_to_predict = False
        print 'TODO: be notified by changes in Cg and Cn'

    def clear_cache_r(self):
        self.clear_cache('Sg','Ug','Vg','Wr','SpI','d')
        self.clear_all()

    def clear_cache_c(self):
        self.clear_cache('Lc','Estar','Se','Ue','Ve','Wc','SpI','d','Ctilde','Cbar')
        self.clear_all()

    #####################
    # Properties
    #####################
    @property
    def G(self):
        return self._G

    @property
    def Cg(self):
        return self._Cg

    @property
    def Cn(self):
        return self._Cn

    @property
    def dim_r(self):
        return self._dim_r

    @property
    def dim_c(self):
        return self._dim_c

    @property
    def rank_r(self):
        return self._rank_r

    @property
    def rank_c(self):
        return self._rank_c

    #####################
    # Setters
    #####################
    @G.setter
    def G(self,value):
        assert value is not None, 'Cov2KronSumLR: Specify G!'
        self._dim_r = value.shape[0]
        self._rank_r = value.shape[1]
        self._G = value 
        self.clear_cache_r()
        self._notify()
        self._notify('row_cov')

    # normal setter for col covars 
    def setColCovars(self, Cn = None, rank = 1):
        assert Cn is not None, 'Cov2KronSumLR: Specify Cn!'
        self._rank_c = rank
        self._dim_c = Cn.dim
        self._Cg = LowRankCov(self._dim_c, rank) 
        self._Cn = Cn
        self.clear_cache_c()
        self._notify()
        self._notify('col_cov')

    #####################
    # Params handling
    #####################
    def setParams(self,params):
        self.Cg.setParams(params[:self.Cg.getNumberParams()])
        self.Cn.setParams(params[self.Cg.getNumberParams():])
        self.clear_cache_c()
        self._notify()
        self._notify('col_cov')

    def getParams(self):
        return sp.concatenate([self.Cg.getParams(),self.Cn.getParams()])

    def _calcNumberParams(self):
        self.n_params = self.Cg.getNumberParams() + self.Cn.getNumberParams() 
        

    #####################
    # Cached
    #####################
    @cached
    def Sg(self):
        Ug, Sgh, Vg = nla.svd(self.G, full_matrices=0)
        self.fill_cache('Ug', Ug)
        self.fill_cache('Vg', Vg)
        return Sgh**2

    @cached
    def Ug(self):
        Ug, Sgh, Vg = nla.svd(self.G, full_matrices=0)
        self.fill_cache('Sg', Sgh**2)
        self.fill_cache('Vg', Vg)
        return Ug

    @cached
    def Vg(self):
        Ug, Sgh, Vg = nla.svd(self.G, full_matrices=0)
        self.fill_cache('Ug', Ug)
        self.fill_cache('Sg', Sgh**2)
        return Vg

    @cached
    def Wr(self):
        return self.Ug().T

    @cached
    def Lc(self):
        return self.Cn.USi2().T

    @cached
    def Lr(self):
        return self.eye(N) 

    @cached
    def Estar(self):
        E = self.Cg.X
        # for a general covariance matrix
        # E = LA.chol(C.K())
        # or based on eigh decomposition if this fails
        return sp.dot(self.Lc(), E)

    @cached
    def Se(self):
        Ue, Seh, Ve = nla.svd(self.Estar(), full_matrices=0)
        self.fill_cache('Ue',Ue)
        self.fill_cache('Ve',Ve)
        return Seh**2

    @cached
    def Ue(self):
        Ue, Seh, Ve = nla.svd(self.Estar(), full_matrices=0)
        self.fill_cache('Se',Seh**2)
        self.fill_cache('Ve',Ve)
        return Ue

    @cached
    def Ve(self):
        Ue, Seh, Ve = nla.svd(self.Estar(), full_matrices=0)
        self.fill_cache('Ue',Ue)
        self.fill_cache('Se',Seh**2)
        return Ve

    @cached
    def Wc(self):
        return self.Ue().T

    @cached
    def SpI(self):
        return sp.kron(1./self.Se(), 1./self.Sg()) + 1 

    @cached
    def d(self):
        return 1./self.SpI()

    def D(self):
        return self.d().reshape((self.rank_r, self.rank_c), order = 'F')

    @cached
    def Ctilde(self, i):
        if i < self.Cg.getNumberParams():
            C = self.Cg.K_grad_i(i)
        else:
            _i = i - self.Cg.getNumberParams()
            C = self.Cn.K_grad_i(_i)
        return sp.dot(self.Lc(), sp.dot(C, self.Lc().T))

    @cached
    def Cbar(self, i):
        return sp.dot(self.Wc(), sp.dot(self.Ctilde(i), self.Wc().T))

    #####################
    # Debug quantities
    #####################
    def L(self):
        assert self.dim <= 5000, 'Cov2kronSum: dimension too big'
        return sp.kron(self.Lc(), sp.eye(self.dim_r))

    def W(self):
        return sp.dot(self.Wc(), self.Wr())

    def R(self):
        assert self.dim <= 5000, 'Cov2kronSum: dimension too big'
        return sp.dot(self.G, self.G.T)

    #####################
    # Overwritten covar_base methods
    #####################
    @cached
    def K(self):
        assert self.dim <= 5000, 'Cov2kronSum: K method not available for matrix with dimensions > 5000'
        rv = sp.kron(self.Cg.K(), self.R()) + sp.kron(self.Cn.K(), sp.eye(self.dim_r))
        return rv

    @cached
    def K_grad_i(self,i):
        assert self.dim <= 5000, 'Cov2kronSum: Kgrad_i method not available for matrix with dimensions > 5000'
        if i < self.Cg.getNumberParams():
            rv= sp.kron(self.Cg.K_grad_i(i), self.R())
        else:
            _i = i - self.Cg.getNumberParams()
            rv = sp.kron(self.Cn.K_grad_i(_i), sp.eye(self.dim_r))
        return rv

    @cached
    def logdet(self):
        rv = sp.sum(sp.log(self.Cn.S())) * self.dim_r
        rv+= sp.log(self.SpI()).sum()
        rv+= sp.log(self.Se()).sum() * self.rank_r
        rv+= sp.log(self.Sg()).sum() * self.rank_c
        return rv

    @cached
    def logdet_grad_i(self,i):
        if i < self.Cg.getNumberParams():
            trR = self.Sg().sum()
            diagR = self.Sg()
        else:
            trR = self.dim_r
            diagR = sp.ones(self.rank_r) 
        rv = self.Ctilde(i).diagonal().sum() * trR
        rv-= (self.d() * sp.kron(sp.diag(self.Cbar(i)), diagR)).sum()
        return rv 

    #####################
    # Debug methods 
    #####################
    def inv_debug(self):
        L = sp.kron(self.Lc(), sp.eye(self.dim_r))
        W = sp.kron(self.Wc(), self.Wr()) 
        WdW = sp.dot(W.T, self.d()[:, sp.newaxis] * W)
        I_WdW = sp.eye(self.dim_c * self.dim_r) - WdW
        return sp.dot(L.T, sp.dot(I_WdW, L))

    @cached
    def logdet_debug(self):
        return 2*sp.log(sp.diag(self.chol())).sum()

    @cached
    def logdet_grad_i_debug(self,i):
        return self.solve(self.K_grad_i(i)).diagonal().sum()
        


