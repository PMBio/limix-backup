import sys
from .covar_base import Covariance
from limix.core.covar import Cov2KronSum
from limix.core.covar import LowRankCov
from hcache import cached
from limix.core.type.exception import TooExpensiveOperationError
from limix.core.utils import my_name
from .util import msg_too_expensive_dim
import scipy as sp
import numpy as np
import scipy.linalg as la
import warnings

import pdb

_MAX_DIM = 5000

class Cov3KronSumLR(Cov2KronSum):
    """
    Covariance class for sum of 3 Kronecker products with one low rank term:
        K = Cr \kron GG.T + Cg \kron R + Cn \kron I
    Notation:
        - dim_c: dimension of col covariances
        - dim_r: dimension of row covariances
        - rank_c: rank of low-rank col covariance
        - rank_r: rank of low-rank row covariance
    """

    def __init__(self, Cg=None, Cn=None, G=None, R=None, rank=1, Cr=None, S_R=None, U_R=None):
        """
        Args:
            Cg:     Limix covariance matrix for Cg (dimension dim_c)
            Cn:     Limix covariance matrix for Cn (dimension dim_c)
            G:      [dim_r, rank_r] numpy covariance matrix for G
            R:      [dim_r, dim_r] numpy semidemidefinite covariance matrix for R
            rank:   rank of column low-rank covariance (default = 1)
            Cr:     Limix covariance matrix for Cr (optional).
                    If not specified, a low-rank covariance matrix is considered
            S_R:    N vector of eigenvalues of R
            U_R:    [N, N] eigenvector matrix of R
        """
        Covariance.__init__(self)
        self._Cr_act = True
        self._Cg_act = True
        self._Cn_act = True
        self.setColCovars(Cg=Cg, Cn=Cn, rank=rank, Cr=None)
        self.setR(R=R, S_R=S_R, U_R=U_R)
        self.G = G
        self.dim = self.dim_c * self.dim_r
        self._calcNumberParams()
        self._use_to_predict = False

    def G_has_changed(self):
        self.clear_cache('G')
        self._notify('G')
        self.clear_all()

    #####################
    # Properties
    #####################
    @property
    def rank_r(self):
        return self._rank_r

    @property
    def rank_c(self):
        return self._rank_c

    @property
    def Cr(self):
        return self._Cr

    @property
    def G(self):
        return self._G

    #####################
    # Setters
    #####################
    @G.setter
    def G(self, value):
        assert value is not None, '%s: Specify G!' % self.__class__.__name__
        assert self.dim_r == value.shape[0], '%s'
        self._G = value
        self._rank_r = value.shape[1]
        self.G_has_changed()

    def setG(self, value):
        self.G = value

    # normal setter for col covars
    def setColCovars(self, Cg=None, Cn=None, rank=1, Cr=None):
        assert Cg is not None, 'Cov2KronSum: Specify Cg!'
        assert Cn is not None, 'Cov2KronSum: Specify Cn!'
        assert Cg.dim==Cn.dim, 'Cov2KronSum: Cg and Cn must have same dimensions!'
        assert Cr is None, '%s: more general covariance matrices are not supported at the moment' % self.__class__.__name__
        if Cr is None:
            Cr = LowRankCov(Cg.dim, rank)
            Cr.setRandomParams()
            self._rank_c = rank
        else:
            self._rank_c = self.Cr.dim()
        self._dim_c = Cg.dim
        self._Cr = Cr
        self._Cg = Cg
        self._Cn = Cn
        self._Cg.register(self.col_covs_have_changed)
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

    def _actindex2index(self, i):
        nCr = self.Cr.getNumberParams()
        nCg = self.Cg.getNumberParams()
        i += nCr * int(not self._Cr_act)
        i += nCg * int(not self._Cg_act)
        return i

    def _index2actindex(self, i):
        nCr = self.Cr.getNumberParams()
        nCg = self.Cg.getNumberParams()
        i -= nCr * int(not self._Cr_act)
        i -= nCg * int(not self._Cg_act)
        return i

    #####################
    # Params handling
    #####################
    def setParams(self,params):
        np_r = self.Cr.getNumberParams()
        np_g = self.Cg.getNumberParams()
        np_n = self.Cn.getNumberParams()

        if len(params) != self.getNumberParams():
            raise ValueError("The number of parameters passed to setParams "
                             "differs from the number of active parameters.")

        self.Cr.setParams(params[:np_r])
        self.Cg.setParams(params[np_r:(np_r + np_g)])
        self.Cn.setParams(params[(np_r + np_g):])

    def getParams(self):
        params = []
        if self._Cr_act:
            params.append(self.Cr.getParams())
        if self._Cg_act:
            params.append(self.Cg.getParams())
        if self._Cn_act:
            params.append(self.Cn.getParams())
        if len(params) == 0:
            return np.array([])
        return sp.concatenate(params)

    def getNumberParams(self):
        np_r = self.Cr.getNumberParams()
        np_g = self.Cg.getNumberParams()
        np_n = self.Cn.getNumberParams()
        nact = (np_r * int(self._Cr_act) + np_g * int(self._Cg_act) +
                np_n * int(self._Cn_act))
        return nact

    def _calcNumberParams(self):
        self.n_params = self.Cr.getNumberParams() + self.Cg.getNumberParams() + self.Cn.getNumberParams()


    #####################
    # Cached
    #####################
    @cached('G')
    def GG(self):
        if self.dim > _MAX_DIM:
            raise TooExpensiveOperationError(msg_too_expensive_dim(my_name(),
                                                                   _MAX_DIM))

        return sp.dot(self.G, self.G.T)

    @cached(['G', 'row_cov'])
    def Wr(self):
        return sp.dot(self.Lr(), self.G)

    @cached('col_cov')
    def Wc(self):
        E = self.Cr.X
        return sp.dot(self.Lc(), E)

    # no need to cached
    def W(self):
        return sp.kron(self.Wc(), self.Wr())

    @cached(['col_cov', 'row_cov', 'G'])
    def DW(self):
        return self.d()[:, sp.newaxis] * self.W()

    @cached(['col_cov', 'row_cov', 'G'])
    def DWt(self):
        return self.DW().reshape((self._dim_r, self.dim_c, self.rank_r * self.rank_c), order = 'F')

    @cached(['col_cov', 'row_cov', 'G'])
    def H_chol(self):
        H = np.tensordot(self.DWt(),self.Wc(),axes=(1,0))
        H = np.transpose(H, (0, 2, 1))
        H = np.tensordot(self.Wr(), H, axes=(0,0))
        H = H.reshape((self.rank_c * self.rank_r, self.rank_c * self.rank_r), order='F')
        H+= np.eye(self.rank_r * self.rank_c)
        return la.cholesky(H).T

    @cached(['col_cov', 'row_cov', 'G'])
    def H_inv(self):
        return la.cho_solve((self.H_chol(), True), sp.eye(self.rank_r * self.rank_c))

    @cached('col_cov')
    def LcGradCrLc(self, i):
        return sp.dot(self.Lc(), sp.dot(self.Cr.K_grad_i(i), self.Lc().T))

    @cached('col_cov')
    def Ctilde(self, i):
        np_r = self.Cr.getNumberParams()
        np_g = self.Cg.getNumberParams()
        if i < np_r:
            r = self.LcGradCrLc(i)
        elif i < (np_r + np_g):
            _i = i - np_r
            r = self.LcGradCgLc(_i)
        else:
            _i = i - np_r - np_g
            r = self.LcGradCnLc(_i)
        return r

    @cached(['row_cov', 'G'])
    def diagWrWr(self):
        return (self.Wr()**2).sum(1)

    @cached(['col_cov', 'row_cov', 'G'])
    def diag_Ctilde_o_Sr(self, i):
        np_r = self.Cr.getNumberParams()
        np_g = self.Cg.getNumberParams()
        if i < np_r:
            r = sp.kron(sp.diag(self.LcGradCrLc(i)), self.diagWrWr())
        elif i < (np_r + np_g):
            _i = i - np_r
            r = sp.kron(sp.diag(self.LcGradCgLc(_i)), self.Sr())
        else:
            _i = i - np_r - np_g
            r = sp.kron(sp.diag(self.LcGradCnLc(_i)), sp.ones(self.dim_r))
        return r

    @cached(['col_cov', 'row_cov', 'G'])
    def WrWrDWt(self):
        R = np.tensordot(self.Wr(), self.DWt(), axes=(0,0))
        R = np.tensordot(self.Wr(), R, axes=(1,0))
        return R

    @cached(['col_cov', 'row_cov', 'G'])
    def SrDWt(self):
        return self.Sr()[:, sp.newaxis, sp.newaxis] * self.DWt()

    @cached(['col_cov', 'row_cov', 'G'])
    def Kbar(self, i):
        # WrDRDWtCtildeWc
        np_r = self.Cr.getNumberParams()
        np_g = self.Cg.getNumberParams()
        if i < np_r:
            RDWt = self.WrWrDWt()
        elif i < (np_r + np_g):
            RDWt = self.SrDWt()
        else:
            RDWt = self.DWt()
        ping = np.tensordot(RDWt, self.Ctilde(i), axes=(1,0))
        pong = self.D()[:, sp.newaxis, :] * ping
        ping = np.tensordot(pong, self.Wc(), axes=(2,0))
        pong = np.tensordot(self.Wr(), ping, axes=(0,0))
        ping = np.transpose(pong, (0,2,1))
        return ping.reshape((self.rank_c * self.rank_r, self.rank_c * self.rank_r),order='F')

    def Sr_X_Ctilde(X, i):
        pass

    #####################
    # Overwritten covar_base methods
    #####################
    @cached(['col_cov', 'row_cov', 'G', 'covar_base'])
    def K(self):
        if self.dim > _MAX_DIM:
            raise TooExpensiveOperationError(msg_too_expensive_dim(my_name(),
                                                                   _MAX_DIM))

        R  = sp.kron(self.Cr.K(), self.GG())
        R += sp.kron(self.Cg.K(), self.R)
        R += sp.kron(self.Cn.K(), sp.eye(self.dim_r))
        return R

    @cached(['col_cov', 'row_cov', 'G', 'covar_base'])
    def K_grad_i(self,i):
        np_r = self.Cr.getNumberParams()
        np_g = self.Cg.getNumberParams()
        np_n = self.Cn.getNumberParams()

        if i >= self.getNumberParams():
            raise ValueError("Trying to retrieve the gradient over a "
                             "parameter that is inactive.")

        if self.dim > _MAX_DIM:
            raise TooExpensiveOperationError(msg_too_expensive_dim(my_name(),
                                                                   _MAX_DIM))

        i = self._actindex2index(i)

        if i < np_r:
            rv= sp.kron(self.Cr.K_grad_i(i), self.GG())
        elif i < (np_r + np_g):
            _i = i - np_r
            rv = sp.kron(self.Cg.K_grad_i(_i), self.R)
        else:
            _i = i - np_r - np_g
            rv = sp.kron(self.Cn.K_grad_i(_i), sp.eye(self.dim_r))
        return rv

    @cached(['col_cov', 'row_cov', 'G', 'covar_base'])
    def logdet(self):
        r = sp.log(self.SpI()).sum()
        r+= sp.sum(sp.log(self.Cn.S())) * self.dim_r
        r+= 2 * sp.log(sp.diag(self.H_chol())).sum()
        return r

    @cached(['col_cov', 'row_cov', 'G', 'covar_base'])
    def logdet_grad_i(self, i):
        r = (self.d() * self.diag_Ctilde_o_Sr(i)).sum()
        r-= (self.H_inv() * self.Kbar(i)).sum()
        return r


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

    # cov = Cov3KronSum(Cg = Cg, Cn = Cn, R = R)
    # cov.setRandomParams()
    #
    # print cov.K()
    # print cov.K_grad_i(0)
