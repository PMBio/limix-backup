import sys
from covar_base import Covariance
from limix.core.type.cached import cached
import scipy as sp
import scipy.linalg as LA
import warnings

import pdb

# should be a child class of combinator
class Cov2KronSum(Covariance):

    def __init__(self, Cg = None, Cn = None, R = None):
        self.setColCovars(Cg, Cn)
        self.R = R
        self.dim = self.dim_c * self.dim_r
        self._calcNumberParams()
        self._use_to_predict = False
        print 'TODO: be notified by changes in Cg and Cn'

    def clear_cache_r(self):
        self.clear_cache('Lr','Sr','SpI','d')
        self.clear_all()

    def clear_cache_c(self):
        self.clear_cache('Cstar','S_Cstar','U_Cstar','SpI','d','Lc','LcGradCgLc','LcGradCnLc')
        self.clear_all()

    #####################
    # Properties
    #####################
    @property
    def R(self):
        return self._R

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

    #####################
    # Setters
    #####################
    @R.setter
    def R(self,value):
        assert value is not None, 'Cov2KronSum: Specify R!'
        self._dim_r = value.shape[0]
        self._R = value 
        self.clear_cache_r()
        self._notify()
        self._notify('row_cov')

    # normal setter for col covars 
    def setColCovars(self, Cg = None, Cn = None):
        assert Cg is not None, 'Cov2KronSum: Specify Cg!'
        assert Cn is not None, 'Cov2KronSum: Specify Cn!'
        assert Cg.dim==Cn.dim, 'Cov2KronSum: Cg and Cn must have same dimensions!'
        self._dim_c = Cg.dim
        self._Cg = Cg 
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
    def Cstar(self):
        return sp.dot(self.Cn.USi2().T,sp.dot(self.Cg.K(),self.Cn.USi2()))

    @cached
    def S_Cstar(self):
        RV,U = LA.eigh(self.Cstar())
        self.fill_cache('U_Cstar',U)
        return RV

    @cached
    def U_Cstar(self):
        S,RV = LA.eigh(self.Cstar())
        self.fill_cache('S_Cstar',S)
        return RV

    @cached
    def SpI(self):
        return sp.kron(self.S_Cstar(),self.Sr())+1

    @cached
    def d(self):
        return 1./self.SpI()

    def D(self):
        return self.d().reshape((self._dim_r, self._dim_c), order = 'F')

    @cached
    def Lc(self):
        return sp.dot(self.U_Cstar().T,self.Cn.USi2().T)

    @cached
    def Sr(self):
        RV,U = LA.eigh(self.R)
        self.fill_cache('Lr',U.T)
        return RV

    @cached
    def Lr(self):
        S,U = LA.eigh(self.R)
        self.fill_cache('Sr',S)
        return U.T

    @cached
    def LcGradCgLc(self, i):
        return sp.dot(self.Lc(), sp.dot(self.Cg.K_grad_i(i), self.Lc().T))

    @cached
    def LcGradCnLc(self, i):
        return sp.dot(self.Lc(), sp.dot(self.Cn.K_grad_i(i), self.Lc().T))

    def Ctilde(self, i):
        if i < self.Cg.getNumberParams():
            r = self.LcGradCgLc(i)
        else:
            _i = i - self.Cg.getNumberParams()
            r = self.LcGradCnLc(_i)
        return r

    def Sr_X_Ctilde(self, X, i):
        if i < self.Cg.getNumberParams():
            SrX = self.Sr()[:, sp.newaxis] * X
            r = sp.dot(SrX, self.LcGradCgLc(i).T)
        else:
            _i = i - self.Cg.getNumberParams()
            r = sp.dot(X, self.LcGradCnLc(_i).T)
        return r

    def diag_Ctilde_o_Sr(self, i):
        if i < self.Cg.getNumberParams():
            r = sp.kron(sp.diag(self.LcGradCgLc(i)), self.Sr())
        else:
            _i = i - self.Cg.getNumberParams()
            r = sp.kron(sp.diag(self.LcGradCnLc(_i)), sp.ones(self.R.shape[0]))
        return r

    #####################
    # Overwritten covar_base methods
    #####################
    @cached
    def K(self):
        assert self.dim <= 5000, 'Cov2kronSum: K method not available for matrix with dimensions > 5000'
        rv = sp.kron(self.Cg.K(), self.R) + sp.kron(self.Cn.K(), sp.eye(self.dim_r))
        return rv

    @cached
    def K_grad_i(self,i):
        assert self.dim <= 5000, 'Cov2kronSum: Kgrad_i method not available for matrix with dimensions > 5000'
        if i < self.Cg.getNumberParams():
            rv= sp.kron(self.Cg.K_grad_i(i), self.R)
        else:
            _i = i - self.Cg.getNumberParams()
            rv = sp.kron(self.Cn.K_grad_i(_i), sp.eye(self.dim_r))
        return rv

    @cached
    def logdet(self):
        return sp.sum(sp.log(self.Cn.S())) * self.R.shape[0] + sp.log(self.SpI()).sum()

    @cached
    def logdet_grad_i(self,i):
        return (self.d() * self.diag_Ctilde_o_Sr(i)).sum()
        


""" Gradients DEPRECATED """

if 0:

    def CstarGrad_g(self,i):
        return sp.dot(self.Cn.USi2().T,sp.dot(self.Cg.Kgrad_param(i),self.Cn.USi2()))

    def CstarGrad_n(self,i):
        RV = sp.dot(self.Cn.USi2grad(i).T,sp.dot(self.Cg.K(),self.Cn.USi2()))
        RV+= sp.dot(self.Cn.USi2().T,sp.dot(self.Cg.K(),self.Cn.USi2grad(i)))
        #RV+= RV.T
        return RV

    def S_CstarGrad_g(self,i):
        return dS_dti(self.CstarGrad_g(i),U=self.U_Cstar())

    def U_CstarGrad_g(self,i):
        return dU_dti(self.CstarGrad_g(i),U=self.U_Cstar(),S=self.S_Cstar())

    def S_CstarGrad_n(self,i):
        return dS_dti(self.CstarGrad_n(i),U=self.U_Cstar())

    def U_CstarGrad_n(self,i):
        return dU_dti(self.CstarGrad_n(i),U=self.U_Cstar(),S=self.S_Cstar())

    def LcGrad_g(self,i):
        return sp.dot(self.U_CstarGrad_g(i).T,self.Cn.USi2().T)

    def LcGrad_n(self,i):
        RV  = sp.dot(self.U_CstarGrad_n(i).T,self.Cn.USi2().T)
        RV += sp.dot(self.U_Cstar().T,self.Cn.USi2grad(i).T)
        return RV

    def Sgrad_g(self,i):
        return sp.kron(self.S_CstarGrad_g(i),self.Sr())

    def Sgrad_n(self,i):
        return sp.kron(self.S_CstarGrad_n(i),self.Sr())

    #####################
    # Debug methods
    #####################
    def inv_debug(self):
        print 'IMPLEMENT ME'

    @cached
    def logdet_debug(self):
        return 2*sp.log(sp.diag(self.chol())).sum()

    @cached
    def logdet_grad_i_debug(self,i):
        return self.solve(self.K_grad_i(i)).diagonal().sum()

if __name__ == '__main__':
    from limix.core.covar import FreeFormCov
    from limix.utils.preprocess import covar_rescale

    # define row caoriance
    dim_r = 10
    X = sp.rand(dim_r, dim_r)
    R = covar_rescale(sp.dot(X,X.T))

    # define col covariances
    dim_c = 3
    Cg = FreeFormCov(dim_c)
    Cn = FreeFormCov(dim_c)

    cov = Cov2KronSum(Cg = Cg, Cn = Cn, R = R)
    cov.setRandomParams()

    print cov.K()
    print cov.K_grad_i(0)



