from .covar_base import Covariance
import pdb
import numpy as np
import scipy as sp
from hcache import Cached, cached

class ACombinatorCov(Covariance):
    """
    Abstract class for combinator of covariances
    """
    def __init__(self):
        Covariance.__init__(self)
        self.covars = []

    #####################
    # Covars handling
    #####################
    def addCovariance(self,covar):
        if self.dim is None:
            self.dim = covar.dim
        else:
            assert covar.dim==self.dim, 'Dimension mismatch.'
        self.covars.append(covar)
        covar.register(self.clear_all)
        self._calcNumberParams()

    def getCovariance(self,i):
        return self.covars[i]

    #####################
    # Params handling
    #####################
    def setParams(self,params):
        istart = 0
        cs = [c for c in self.covars if c.getNumberParams() > 0]
        for c in cs:
            n = c.getNumberParams()
            istop = istart + n
            c.setParams(params[istart:istop])
            istart = istop

    def getParams(self):
        istart = 0
        params = sp.zeros(self.getNumberParams())
        cs = [c for c in self.covars if c.getNumberParams() > 0]
        for c in cs:
            istop = istart + c.getNumberParams()
            params[istart:istop] = c.getParams()
            istart = istop
        return params

    def getNumberParams(self):
        return np.sum([c.getNumberParams() for c in self.covars])

    ####################
    # Predictions
    ####################
    @property
    def use_to_predict(self):
        r = False
        for i in range(len(self.covars)):
            r = r or self.getCovariance(i).use_to_predict
        return r

    @use_to_predict.setter
    def use_to_predict(self,value):
        raise NotImplementedError("This method is only implemented for single"
                                  " covariance terms.")

    ####################
    # Interpretable Params and Ste
    ####################
    def getInterParams(self):
        istart = 0
        interParams = sp.zeros(self.getNumberParams())
        for i in range(len(self.covars)):
            istop = istart + self.getCovariance(i).getNumberParams()
            params[istart:istop] = self.getCovariance(i).getInterParams()
            istart = istop
        return params

    def setFIinv(self, value):
        self._FIinv = value
        istart = 0
        for i in range(len(self.covars)):
            istop = istart + self.getCovariance(i).getNumberParams()
            self.getCovariance(i).setFIinv(value[istart:istop][:,istart:istop])
            istart = istop


