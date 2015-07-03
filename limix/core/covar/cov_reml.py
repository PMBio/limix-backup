import sys
from limix.core.type.cached import cached
import scipy as sp
from covar_base import Covariance
import pdb
import scipy.linalg as LA
import warnings

import logging as LG

class cov_reml(Covariance):
    """
    REML covariance used by GP for restricted maximum likelihood
    """
    def __init__(self,gp):
        Covariance.__init__(self)
        self.gp = gp
        gp.register(self.clear_all)
        self.dim = gp.mean.b.shape[0]
        # self._calcNumberParams()

    def clear_all(self):
        self.clear_cache('default')

    #####################
    # Params handling
    #####################
    def setParams(self,params):
        warnings.warn('Read-only covariance type')

    def getParams(self,params):
        return self.gp.covar.getParams()

    # def _calcNumberParams(self):
    #     self.n_params = self.gp.covar.getNumberParams()

    #####################
    # Cached
    #####################
    @cached
    def K(self):
        return self.gp.Areml_K()

    @cached
    def K_grad_i(self,i):
        return self.gp.Areml_K_grad_i(i)
