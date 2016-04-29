import sys
import numpy as np
from .covar_base import Covariance
from hcache import cached
from limix.core.type.exception import TooExpensiveOperationError
from limix.core.utils import my_name
from .util import msg_too_expensive_dim
import scipy as sp
import scipy.linalg as LA
import warnings

import pdb

_MAX_DIM = 5000

# should be a child class of combinator
class Cov2KronSum(Covariance):
    """
    Covariance class for sum of two Kronecker products
    K = Cg \kron R + Cn \kron I
    Notation:
        - dim_c: dimension of col covariances
        - dim_r: dimension of row covariances
    """

    def __init__(self, Cg = None, Cn = None, R = None, S_R = None, U_R = None):
        """
        Args:
            Cg:     Limix covariance matrix for Cg (dimension dim_c)
            Cn:     Limix covariance matrix for Cn (dimension dim_c)
            R:      [dim_r, dim_r] numpy semidemidefinite covariance matrix for R
                    In alternative to R, S_R and U_R can be specified.
            S_R:    N vector of eigenvalues of R
            U_R:    [N, N] eigenvector matrix of R
        """
        Covariance.__init__(self)
        self._Cg_act = True
        self._Cn_act = True
        self.setColCovars(Cg, Cn)
        self.setR(R=R, S_R=S_R, U_R=U_R)
        self.dim = self.dim_c * self.dim_r
        self._use_to_predict = False

    def col_covs_have_changed(self):
        self.clear_cache('col_cov')
        self._notify('col_cov')
        self.clear_all()

    def R_has_changed(self):
        self.clear_cache('row_cov')
        self._notify('row_cov')
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
    def setR(self, R=None, S_R=None, U_R=None):
        RnotNone  = R is not None
        SUnotNone = S_R is not None and U_R is not None
        assert RnotNone or SUnotNone, 'Either R or S_R and U_R must be specified.'
        if RnotNone:
            self._dim_r = R.shape[0]
            self._R = R
            self.R_has_changed()
        if SUnotNone:
            #TODO: assert dimensions
            self._dim_r = S_R.shape[0]
            self.R_has_changed()
            self.fill_cache('Lr', U_R.T)
            self.fill_cache('Sr', S_R)

    def setColCovars(self, Cg = None, Cn = None):
        assert Cg is not None, 'Cg has to be specified.'
        assert Cn is not None, 'Cn has to be specified.'
        assert Cg.dim==Cn.dim, 'Cg and Cn must have the same dimensions.'
        self._dim_c = Cg.dim
        self._Cg = Cg
        self._Cn = Cn
        self._Cg.register(self.col_covs_have_changed)
        self._Cn.register(self.col_covs_have_changed)
        self.col_covs_have_changed()

    #####################
    # Activation handling
    #####################
    @property
    def act_Cg(self):
        return self._Cg_act

    @act_Cg.setter
    def act_Cg(self, act):
        self._Cg_act = bool(act)
        self._notify()

    @property
    def act_Cn(self):
        return self._Cn_act

    @act_Cn.setter
    def act_Cn(self, act):
        self._Cn_act = bool(act)
        self._notify()

    #####################
    # Params handling
    #####################
    def setParams(self, params):
        nCg = self.Cg.getNumberParams()
        nCn = self.Cn.getNumberParams()
        nact = nCg * int(self._Cg_act) + nCn * int(self._Cn_act)

        if len(params) != nact:
            raise ValueError("The number of parameters passed to setParams "
                             "differs from the number of active parameters.")

        self.Cg.setParams(params[:nCg])
        self.Cn.setParams(params[nCg:])

    def getParams(self):
        params = []
        if self._Cg_act:
            params.append(self.Cg.getParams())
        if self._Cn_act:
            params.append(self.Cn.getParams())
        if len(params) == 0:
            return np.array([])
        return sp.concatenate(params)

    def getNumberParams(self):
        return (int(self._Cg_act) * self.Cg.getNumberParams() +
                int(self._Cn_act) * self.Cn.getNumberParams())

    def _actindex2index(self, i):
        nCg = self.Cg.getNumberParams()
        i += nCg * int(not self._Cg_act)
        return i

    def _index2actindex(self, i):
        nCg = self.Cg.getNumberParams()
        i -= nCg * int(not self._Cg_act)
        return i

    #####################
    # Cached
    #####################
    @cached('col_cov')
    def Cstar(self):
        return sp.dot(self.Cn.USi2().T,sp.dot(self.Cg.K(),self.Cn.USi2()))

    @cached('col_cov')
    def S_Cstar(self):
        RV,U = LA.eigh(self.Cstar())
        self.fill_cache('U_Cstar',U)
        return RV

    @cached('col_cov')
    def U_Cstar(self):
        S,RV = LA.eigh(self.Cstar())
        self.fill_cache('S_Cstar',S)
        return RV

    @cached(['col_cov', 'row_cov'])
    def SpI(self):
        return sp.kron(self.S_Cstar(),self.Sr())+1

    @cached(['col_cov', 'row_cov'])
    def d(self):
        return 1./self.SpI()

    @cached(['col_cov', 'row_cov'])
    def D(self):
        return self.d().reshape((self._dim_r, self._dim_c), order = 'F')

    @cached('col_cov')
    def Lc(self):
        return sp.dot(self.U_Cstar().T,self.Cn.USi2().T)

    @cached('row_cov')
    def Sr(self):
        RV, U = LA.eigh(self.R, check_finite=False)
        self.fill_cache('Lr',U.T)
        return RV

    @cached('row_cov')
    def Lr(self):
        S,U = LA.eigh(self.R)
        self.fill_cache('Sr',S)
        return U.T

    @cached('col_cov')
    def LcGradCgLc(self, i):
        return sp.dot(self.Lc(), sp.dot(self.Cg.K_grad_i(i), self.Lc().T))

    @cached('col_cov')
    def LcGradCnLc(self, i):
        return sp.dot(self.Lc(), sp.dot(self.Cn.K_grad_i(i), self.Lc().T))

    @cached('col_cov')
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
            r = sp.kron(sp.diag(self.LcGradCnLc(_i)), sp.ones(self.dim_r))
        return r

    @cached(['row_cov', 'col_cov'])
    def L(self):
        if self.dim > _MAX_DIM:
            raise TooExpensiveOperationError(msg_too_expensive_dim(my_name(),
                                                                   _MAX_DIM))

        return sp.kron(self.Lc(), self.Lr())

    #####################
    # Overwritten covar_base methods
    #####################
    @cached(['row_cov', 'col_cov', 'covar_base'])
    def K(self):
        if self.dim > _MAX_DIM:
            raise TooExpensiveOperationError(msg_too_expensive_dim(my_name(),
                                                                   _MAX_DIM))

        rv = sp.kron(self.Cg.K(), self.R) + sp.kron(self.Cn.K(), sp.eye(self.dim_r))
        return rv

    @cached(['row_cov', 'col_cov', 'covar_base'])
    def K_grad_i(self,i):
        nCg = self.Cg.getNumberParams()

        if i >= self.getNumberParams():
            raise ValueError("Trying to retrieve the gradient over a "
                             "parameter that is inactive.")

        if self.dim > _MAX_DIM:
            raise TooExpensiveOperationError(msg_too_expensive_dim(my_name(),
                                                                   _MAX_DIM))

        i = self._actindex2index(i)

        if i < nCg:
            rv= sp.kron(self.Cg.K_grad_i(i), self.R)
        else:
            _i = i - nCg
            rv = sp.kron(self.Cn.K_grad_i(_i), sp.eye(self.dim_r))
        return rv

    @cached(['row_cov', 'col_cov', 'covar_base'])
    def logdet(self):
        return sp.sum(sp.log(self.Cn.S())) * self.dim_r + sp.log(self.SpI()).sum()

    @cached(['row_cov', 'col_cov', 'covar_base'])
    def logdet_grad_i(self,i):
        if i >= self.getNumberParams():
            raise ValueError("Trying to retrieve the gradient over a "
                             "parameter that is inactive.")
        i = self._actindex2index(i)
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

    print((cov.K()))
    print((cov.K_grad_i(0)))
