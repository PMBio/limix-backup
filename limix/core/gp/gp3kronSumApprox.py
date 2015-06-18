import sys
import pdb
import numpy as NP
import scipy as SP
import scipy.linalg as LA
import scipy.sparse.linalg as SLA
import scipy.sparse as SS
import time as TIME
import copy

from gp_base import GP
from limix.core.covar.cov3kronSum import cov3kronSum
from limix.core.covar import freeform
from limix.core.covar import Covariance
from limix.core.utils import assert_type
from limix.core.utils import assert_subtype

from limix.core.utils import assert_type_or_list_type


class gp3kronSumApprox(GP):

    def __init__(self, Y, C1, C2, Cn, R1, R2, tol=1E-3):
        """
        Y:      Phenotype matrix
        C1:     LIMIX trait-to-trait covariance for region contribution
        C2:     LIMIX trait-to-trait covariance for genetic contribution
        Cn:     LIMIX trait-to-trait covariance for noise
        R2:     Matrix for fixed sample-to-sample covariance function
        """

        assert_type(Y, NP.ndarray, 'Y')
        assert_subtype(C1, Covariance, 'C1')
        assert_subtype(C2, Covariance, 'C2')
        assert_subtype(Cn, Covariance, 'Cn')
        assert_type(R1, NP.ndarray, 'R1')
        assert_type(R2, NP.ndarray, 'R2')

        # time
        self.time = SP.zeros(20)

        # dimensions
        self.N, self.P = Y.shape

        # pheno
        self.Y = Y
        self.y = SP.reshape(self.Y, (self.N*self.P), order='F')

        # opt
        self.tol = tol

        # covars
        self.K = cov3kronSum(C1=C1, C2=C2, Cn=Cn, R1=R1, R2=R2)

        # init cache and params
        self.cache = {}
        self.params = None

    def getParams(self):
        """
        get hper parameters
        """
        params = {}
        params['C1'] = self.K.C1.getParams()
        params['C2'] = self.K.C2.getParams()
        params['Cn'] = self.K.Cn.getParams()
        return params

    def setParams(self, params):
        """
        set hper parameters
        """
        self.params = params
        self.updateParams()

    def updateParams(self):
        """
        update parameters
        """
        params = SP.concatenate([self.params['C1'], self.params['C2'],
                                 self.params['Cn']])
        self.K.setParams(params)

    def _update_cache(self):
        """
        Update cache
        """
        cov_params_have_changed = (self.K.C1.params_have_changed or
                                   self.K.C2.params_have_changed or
                                   self.K.Cn.params_have_changed)

        if 'KiY' not in self.cache or self.R2_has_changed:
            self.cache['KiY0'] = 1e-3*SP.randn(self.N, self.P)

        if (cov_params_have_changed or self.R2_has_changed or
                self.R1_has_changed):
            start = TIME.time()
            self.cache['KiY'], self.cache['KiY0'] =\
                self.K.solve(self.Y, X0=self.cache['KiY0'], tol=self.tol)
            self.time[2] += TIME.time() - start

        self.R2_has_changed = False
        self.R1_has_changed = False
        self.K.C1.params_have_changed = False
        self.K.C2.params_have_changed = False
        self.K.Cn.params_have_changed = False

    def LML(self, params=None, *kw_args):
        """
        calculate LML
        """
        if params is not None:
            self.setParams(params)
        self._update_cache()

        # 1. constant term
        lml = self.N*self.P*SP.log(2*SP.pi)
        # 2. log det
        lml += self.K.logdet_bound()
        # 3. quadratic term
        lml += (self.Y*self.cache['KiY']).sum()
        lml *= 0.5

        return lml

    def LML_debug(self, params=None, *kw_args):
        """
        calculate LML naively
        """
        assert self.N*self.P < 10000, 'gp3kronSum:: N*P>=10000'

        if params is not None:
            self.setParams(params)
        self._update_cache()

        start = TIME.time()
        y = SP.reshape(self.Y, (self.N*self.P), order='F')

        cholK = LA.cholesky(self.K.K())
        Kiy = LA.cho_solve((cholK, False), y)

        lml = y.shape[0]*SP.log(2*SP.pi)
        lml += 2*SP.log(SP.diag(cholK)).sum()
        lml += SP.dot(y, Kiy)
        lml *= 0.5
        self.time[7] = TIME.time() - start

        return lml

    def LMLgrad(self, params=None, **kw_args):
        """
        LML gradient
        """
        if params is not None:
            self.setParams(params)
        self._update_cache()
        RV = {}
        covars = ['C1', 'C2', 'Cn']
        for covar in covars:
            RV[covar] = self._LMLgrad_covar(covar)
        return RV

    def _LMLgrad_covar(self, covar, **kw_args):
        """
        calculates LMLgrad for covariance parameters
        """
        # preprocessing
        start = TIME.time()
        if covar == 'C1':
            n_params = self.K.C1.getNumberParams()
            RKiY = SP.dot(self.K.R1, self.cache['KiY'])
        elif covar == 'C2':
            n_params = self.K.C2.getNumberParams()
            RKiY = SP.dot(self.K.R2, self.cache['KiY'])
        elif covar == 'Cn':
            n_params = self.K.Cn.getNumberParams()
            RKiY = self.cache['KiY']
        self.time[3] += TIME.time() - start

        # fill gradient vector
        RV = SP.zeros(n_params)
        for i in range(n_params):
            start = TIME.time()
            if covar == 'C1':
                C = self.K.C1.Kgrad_param(i)
                logdetGrad = self.K.logdet_grad_1(i)
            elif covar == 'C2':
                C = self.K.C2.Kgrad_param(i)
                logdetGrad = self.K.logdet_grad_2(i)
            elif covar == 'Cn':
                C = self.K.Cn.Kgrad_param(i)
                logdetGrad = self.K.logdet_grad_n(i)
            self.time[4] += TIME.time() - start

            # 1. der of logdet grad
            start = TIME.time()
            RV[i] = logdetGrad
            self.time[5] += TIME.time() - start

            # 2. der of quad term
            start = TIME.time()
            RV[i] -= SP.sum(self.cache['KiY']*SP.dot(RKiY, C.T))
            self.time[6] += TIME.time() - start

            RV[i] *= 0.5

        return RV
