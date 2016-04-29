import sys
import numpy as np
from .covar_base import Covariance
from limix.core.covar import LowRankCov
from hcache import cached
from limix.core.type.exception import TooExpensiveOperationError
from limix.core.utils import my_name
from .util import msg_too_expensive_dim
from limix.utils.svd_utils import svd_reduce
import scipy as sp
import scipy.linalg as la
import numpy.linalg as nla
import warnings

import pdb

_MAX_DIM = 5000

class Cov2KronSumLR(Covariance):
    """
    Covariance class for sum of 2 Kronecker products with one low rank term:
        K = Cr \kron GG.T + Cn \kron I
    Notation:
        - dim_c: dimension of col covariances
        - dim_r: dimension of row covariances
        - rank_c: rank of low-rank col covariance
        - rank_r: rank of low-rank row covariance
    """

    def __init__(self, Cn = None, G = None, rank = 1):
        """
        Args:
            Cn:     Limix covariance matrix for Cn (dimension dim_c)
            G:      [dim_r, rank_r] numpy covariance matrix for G
            rank:   rank of column low-rank covariance (default = 1)
        """
        Covariance.__init__(self)
        self._Cr_act = True
        self._Cn_act = True
        self.setColCovars(Cn, rank = rank)
        self.G = G
        self.dim = self.dim_c * self.dim_r
        self._use_to_predict = False

    def G_has_changed(self):
        self.clear_cache('row_cov')
        self._notify('row_cov')
        self.clear_all()

    def col_covs_have_changed(self):
        self.clear_cache('col_cov')
        self._notify('col_cov')
        self.clear_all()

    #####################
    # Properties
    #####################
    @property
    def G(self):
        return self._G

    @property
    def Cr(self):
        return self._Cr

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
        assert value is not None, 'G cannot be set to None.'
        self._dim_r = value.shape[0]
        # perform svd on G
        # excludes eigh < 1e-8 and recalcs G
        _value, U, S, V = svd_reduce(value)
        self._rank_r = _value.shape[1]
        self._G  = _value
        self._Ug = U
        self._Sg = S
        self._Vg = V
        self.G_has_changed()

    # normal setter for col covars
    def setColCovars(self, Cn = None, rank = 1):
        assert Cn is not None, 'Cn has to be specified.'
        self._rank_c = rank
        self._dim_c = Cn.dim
        self._Cr = LowRankCov(self._dim_c, rank)
        self._Cn = Cn
        # register
        self._Cr.register(self.col_covs_have_changed)
        self._Cn.register(self.col_covs_have_changed)
        self.col_covs_have_changed()

    #####################
    # Activation handling
    #####################
    @property
    def act_Cr(self):
        return self._Cr_act

    @act_Cr.setter
    def act_Cr(self, act):
        self._Cr_act = bool(act)
        self._notify()

    @property
    def act_Cn(self):
        return self._Cn_act

    @act_Cn.setter
    def act_Cn(self, act):
        self._Cn_act = bool(act)
        self._notify()

    def _actindex2index(self, i):
        nCr = self.Cr.getNumberParams()
        i += nCr * int(not self._Cr_act)
        return i

    def _index2actindex(self, i):
        nCr = self.Cr.getNumberParams()
        i -= nCr * int(not self._Cr_act)
        return i


    #####################
    # Params handling
    #####################
    def setParams(self, params):
        nCr = int(self._Cr_act) * self.Cr.getNumberParams()
        nCn = int(self._Cn_act) * self.Cn.getNumberParams()

        if len(params) != self.getNumberParams():
            raise ValueError("The number of parameters passed to setParams "
                             "differs from the number of active parameters.")
        if self._Cr_act:
            self.Cr.setParams(params[:nCr])
        if self._Cn_act:
            self.Cn.setParams(params[nCr:])

    def getParams(self):
        params = []
        if self._Cr_act:
            params.append(self.Cr.getParams())
        if self._Cn_act:
            params.append(self.Cn.getParams())
        if len(params) == 0:
            return np.array([])
        return sp.concatenate(params)

    def getNumberParams(self):
        return (int(self._Cr_act) * self.Cr.getNumberParams() +
                int(self._Cn_act) * self.Cn.getNumberParams())


    #####################
    # Cached
    #####################
    def Sg(self):
        return self._Sg

    def Ug(self):
        return self._Ug

    def Vg(self):
        return self._Vg

    @cached('row_cov')
    def trSg(self):
        return self.Sg().sum()

    @cached('row_cov')
    def logdetSg(self):
        return sp.log(self.Sg()).sum()

    @cached('row_cov')
    def Wr(self):
        return self.Ug().T

    @cached('col_cov')
    def Lc(self):
        return self.Cn.USi2().T

    @cached
    def Lr(self):
        return self.eye(N)

    @cached('col_cov')
    def Estar(self):
        E = self.Cr.X
        # for a general covariance matrix
        # E = LA.chol(C.K())
        # or based on eigh decomposition if this fails
        return sp.dot(self.Lc(), E)

    @cached('col_cov')
    def Se(self):
        Ue, Seh, Ve = nla.svd(self.Estar(), full_matrices=0)
        self.fill_cache('Ue',Ue)
        self.fill_cache('Ve',Ve)
        return Seh**2

    @cached('col_cov')
    def Ue(self):
        Ue, Seh, Ve = nla.svd(self.Estar(), full_matrices=0)
        self.fill_cache('Se',Seh**2)
        self.fill_cache('Ve',Ve)
        return Ue

    @cached('col_cov')
    def Ve(self):
        Ue, Seh, Ve = nla.svd(self.Estar(), full_matrices=0)
        self.fill_cache('Ue',Ue)
        self.fill_cache('Se',Seh**2)
        return Ve

    @cached('col_cov')
    def Wc(self):
        return self.Ue().T

    @cached(['row_cov', 'col_cov'])
    def SpI(self):
        return sp.kron(1./self.Se(), 1./self.Sg()) + 1

    @cached(['row_cov', 'col_cov'])
    def d(self):
        return 1./self.SpI()

    @cached(['row_cov', 'col_cov'])
    def D(self):
        return self.d().reshape((self.rank_r, self.rank_c), order = 'F')

    @cached(['col_cov'])
    def Ctilde(self, i):
        if i < self.Cr.getNumberParams():
            C = self.Cr.K_grad_i(i)
        else:
            _i = i - self.Cr.getNumberParams()
            C = self.Cn.K_grad_i(_i)
        return sp.dot(self.Lc(), sp.dot(C, self.Lc().T))

    @cached(['col_cov'])
    def LcCtildeLc(self, i):
        return sp.dot(self.Lc().T, sp.dot(self.Ctilde(i), self.Lc()))

    @cached(['col_cov'])
    def Cbar(self, i):
        return sp.dot(self.Wc(), sp.dot(self.Ctilde(i), self.Wc().T))

    #####################
    # Debug quantities
    #####################
    def L(self):
        if self.dim > _MAX_DIM:
            raise TooExpensiveOperationError(msg_too_expensive_dim(my_name(),
                                                                   _MAX_DIM))

        return sp.kron(self.Lc(), sp.eye(self.dim_r))

    def W(self):
        return sp.dot(self.Wc(), self.Wr())

    def R(self):
        if self.dim > _MAX_DIM:
            raise TooExpensiveOperationError(msg_too_expensive_dim(my_name(),
                                                                   _MAX_DIM))

        return sp.dot(self.G, self.G.T)

    #####################
    # Overwritten covar_base methods
    #####################
    @cached(['row_cov', 'col_cov', 'covar_base'])
    def K(self):
        if self.dim > _MAX_DIM:
            raise TooExpensiveOperationError(msg_too_expensive_dim(my_name(),
                                                                   _MAX_DIM))

        rv = sp.kron(self.Cr.K(), self.R()) + sp.kron(self.Cn.K(), sp.eye(self.dim_r))
        return rv

    @cached(['row_cov', 'col_cov', 'covar_base'])
    def K_grad_i(self,i):
        n = self.getNumberParams()

        if i >= n:
            raise ValueError("Trying to retrieve the gradient over a "
                             "parameter that is inactive.")

        if self.dim > _MAX_DIM:
            raise TooExpensiveOperationError(msg_too_expensive_dim(my_name(),
                                                                   _MAX_DIM))

        i = self._actindex2index(i)

        nCr = self.Cr.getNumberParams()

        if i < nCr:
            rv= sp.kron(self.Cr.K_grad_i(i), self.R())
        else:
            _i = i - nCr
            rv = sp.kron(self.Cn.K_grad_i(_i), sp.eye(self.dim_r))
        return rv

    @cached(['row_cov', 'col_cov', 'covar_base'])
    def logdet(self):
        rv = sp.sum(sp.log(self.Cn.S())) * self.dim_r
        rv+= sp.log(self.SpI()).sum()
        rv+= sp.log(self.Se()).sum() * self.rank_r
        rv+= self.logdetSg() * self.rank_c
        return rv

    @cached(['row_cov', 'col_cov', 'covar_base'])
    def logdet_grad_i(self,i):

        if i >= self.getNumberParams():
            raise ValueError("Trying to retrieve the gradient over a "
                             "parameter that is inactive.")
        i = self._actindex2index(i)

        if i < self.Cr.getNumberParams():
            trR = self.trSg()
            diagR = self.Sg()
        else:
            trR = self.dim_r
            diagR = sp.ones(self.rank_r)
        rv = self.Ctilde(i).diagonal().sum() * trR
        rv-= (self.d() * sp.kron(sp.diag(self.Cbar(i)), diagR)).sum()
        return rv
