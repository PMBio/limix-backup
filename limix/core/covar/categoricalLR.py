import sys
import numpy as np
from .covar_base import Covariance
from .freeform import FreeFormCov
from .diagonal import DiagonalCov 
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

class CategoricalLR(Covariance):

    def __init__(self, Cr=None, G=None, Ie=None):
        """
        Args:
            Cr:     Limix covariance matrix for Cr (dimension 2) 
                        must be freeform, lowrank or block
            G:      [dim, rank] numpy covariance matrix for G
            Ie:     environment labels Ie \in {0,1}^dim
        """
        Covariance.__init__(self)
        self._Cr_act = True
        self._Cn_act = True
        self.setColCovars(Cr)
        self.G = G
        self.Ie = Ie
        self._use_to_predict = False

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
    def dim(self):
        return self._dim

    #####################
    # Setters
    #####################
    @G.setter
    def G(self,value):
        assert value is not None, 'G cannot be set to None.'
        self._dim = value.shape[0]
        # perform svd on G
        # excludes eigh < 1e-8 and recalcs G
        _value, U, S, V = svd_reduce(value)
        self._G  = _value
        self._Ug = U
        self._Sg = S
        self._Vg = V
        self.clear_all()

    # normal setter for col covars
    def setColCovars(self, Cr = None, rank = 1):
        assert Cr is not None, 'Cr has to be specified.'
        self._Cr = Cr 
        self._Cn = DiagonalCov(2)
        # register
        self._Cr.register(self.clear_all)
        self._Cn.register(self.clear_all)
        self.clear_all()

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
    # Non-cached methods
    #####################
    def solve(self, M):
        DiM = self.d_inv()[:, sp.newaxis] * M
        DiWHiWDiM = sp.dot(self.DiW(), sp.dot(self.H_inv(), sp.dot(self.W().T, DiM)))
        RV = DiM - DiWHiWDiM
        return RV

    def K_grad_i_dot(self, M, i):
        if i < self.Cr.getNumberParams():
            R = sp.dot(self.W(), sp.dot(self.W_grad_i(i).T, M))
            R+= sp.dot(self.W_grad_i(i), sp.dot(self.W().T, M))
        else:
            R = self.d_grad_i(i-self.Cr.getNumberParams())[:, sp.newaxis] * M
        return R
        

    #####################
    # Cached
    #####################
    @cached('covar_base')
    def W(self):
        RV = sp.zeros((self.dim, self.Cr.X.shape[1]*self.G.shape[1]))
        RV[self.Ie] = sp.kron(self.Cr.X[0], self.G[self.Ie])
        RV[~self.Ie] = sp.kron(self.Cr.X[1], self.G[~self.Ie])
        return RV 

    @cached('covar_base')
    def d_inv(self):
        RV = sp.zeros(self.dim)
        RV[self.Ie] = 1. / self.Cn.variance[0]
        RV[~self.Ie] = 1. / self.Cn.variance[1]
        return RV 

    @cached('covar_base')
    def DiW(self):
        return self.d_inv()[:, sp.newaxis] * self.W()

    @cached('covar_base')
    def WDiW(self):
        return sp.dot(self.W().T, self.DiW())

    @cached('covar_base')
    def H_chol(self):
        return la.cholesky(self.WDiW()+sp.eye(self.WDiW().shape[0])).T

    @cached('covar_base')
    def H_inv(self):
        return la.cho_solve((self.H_chol(), True), sp.eye(self.W().shape[1]))

    @cached('covar_base')
    def W_grad_i(self, i):
        RV = sp.zeros((self.dim, self.Cr.X.shape[1]*self.G.shape[1]))
        RV[self.Ie] = sp.kron(self.Cr.Xgrad(i)[0], self.G[self.Ie])
        RV[~self.Ie] = sp.kron(self.Cr.Xgrad(i)[1], self.G[~self.Ie])
        return RV 

    @cached('covar_base')
    def d_grad_i(self, i):
        RV = sp.zeros(self.dim)
        if i==0:
            RV[self.Ie] = self.Cn.variance[0]
        elif i==1:
            RV[~self.Ie] = self.Cn.variance[1]
        return RV 

    #####################
    # Overwritten covar_base methods
    #####################
    @cached('covar_base')
    def K(self):
        print('DO NOT USE ME')
        return sp.dot(self.W(), self.W().T) + sp.diag(self.d_inv()**(-1))

    @cached('covar_base')
    def logdet(self):
        return 2*sp.log(sp.diag(self.H_chol())).sum() - sp.log(self.d_inv()).sum()

    @cached('covar_base')
    def logdet_grad_i(self,i):
        if i >= self.getNumberParams():
            raise ValueError("Trying to retrieve the gradient over a "
                             "parameter that is inactive.")
        i = self._actindex2index(i)
        if i < self.Cr.getNumberParams():
            diag = 2*(self.W()*self.W_grad_i(i)).sum(1)
            #WDiW_grad_i = sp.dot(self.DiW().T, self.W_grad_i(i))
            #WDiW_grad_i+= WDiW_grad_i.T 
            #WDidKDiW = sp.dot(WDiW_grad_i, self.WDiW())
            _WDidKDiW = sp.dot(sp.dot(self.DiW().T, self.W_grad_i(i)), self.WDiW())
            WDidKDiW = _WDidKDiW + _WDidKDiW.T 
        else:
            diag = self.d_grad_i(i-self.Cr.getNumberParams())
            WDidKDiW = sp.dot(self.DiW().T, diag[:, sp.newaxis]*self.DiW())
        rv = sp.dot(self.d_inv(), diag)
        rv-= (self.H_inv() * WDidKDiW).sum()
        return rv

    @cached('covar_base')
    def K_grad_i(self,i):
        print('DO NOT USE ME')
        if i >= self.getNumberParams():
            raise ValueError("Trying to retrieve the gradient over a "
                             "parameter that is inactive.")
        i = self._actindex2index(i)
        if i < self.Cr.getNumberParams():
            _R = sp.dot(self.W(), self.W_grad_i(i).T)
            R = _R + _R.T
        else:
            R = sp.diag(self.d_grad_i(i-self.Cr.getNumberParams()))
        return R

if __name__=='__main__':

    import pylab as pl
    pl.ion()

    N = 100; S = 10
    G = 1. * (sp.rand(N, S)<0.2)
    G-= G.mean(0); G/= G.std(0); G /= sp.sqrt(S)
    Ie = sp.rand(N)<0.5
    Cr = FreeFormCov(2, jitter=0.)
    C = CategoricalLR(Cr, G, Ie)
    C.setRandomParams()

    # calc WW using W
    WW = sp.dot(C.W(), C.W().T)
    # calc WW directly
    Cr_big = sp.zeros((N,N))
    Cr_big[sp.ix_(Ie, Ie)]  = Cr.K()[0,0]
    Cr_big[sp.ix_(Ie, ~Ie)] = Cr.K()[0,1]
    Cr_big[sp.ix_(~Ie, Ie)] = Cr.K()[1,0]
    Cr_big[sp.ix_(~Ie, ~Ie)]= Cr.K()[1,1]
    WW1 = Cr_big * sp.dot(G, G.T)
    print(('WW:', ((WW1-WW)**2).mean()))

    pdb.set_trace()
    
    for i in range(3):
        # calc D(WW)
        _WW_grad = sp.dot(C.W_grad_i(i), C.W().T)
        WW_grad = _WW_grad + _WW_grad.T
        # calc WW directly
        Cr_big = sp.zeros((N,N))
        Cr_big[sp.ix_(Ie, Ie)]  = Cr.K_grad_i(i)[0,0]
        Cr_big[sp.ix_(Ie, ~Ie)] = Cr.K_grad_i(i)[0,1]
        Cr_big[sp.ix_(~Ie, Ie)] = Cr.K_grad_i(i)[1,0]
        Cr_big[sp.ix_(~Ie, ~Ie)]= Cr.K_grad_i(i)[1,1]
        WW1_grad = Cr_big * sp.dot(G, G.T)
        print(('WWgrad %d:' % i, ((WW1_grad-WW_grad)**2).mean()))

    # inverse
    y = sp.randn(N, 1)
    Kiy1 = C.solve(y)
    Kiy2 = sp.dot(C.inv(), y)
    print(((Kiy1-Kiy2)**2).mean())

    # logdet
    ld1 = C.logdet()
    ld2 = sp.log(C.S()).sum() 
    print((ld1-ld2))

    # logdet_grad_i
    for i in range(C.getNumberParams()):
        ldg1 = C.logdet_grad_i(i)
        ldg2 = sp.dot(C.inv(), C.K_grad_i(i)).diagonal().sum()
        print(('logdet grad %d:' % i, ldg1-ldg2))

    # K_grad_dot
    M = sp.randn(N, 1)
    for i in range(C.getNumberParams()):
        DKM = C.K_grad_i_dot(M, i) 
        DKM1 = sp.dot(C.K_grad_i(i), M)
        print(('Kgrad%d.dot:' % i, ((DKM-DKM1)**2).mean()))

    pdb.set_trace()

