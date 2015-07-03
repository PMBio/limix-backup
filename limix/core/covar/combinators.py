from covar_base import Covariance
import pdb
import numpy as np
import scipy as SP
from limix.core.type.cached import Cached, cached

class SumCov(Covariance):
    """
    Sum of multiple covariance matrices.
    The number of paramteters is the sum of the parameters of the single covariances.
    """

    def __init__(self,*covars):
        """
        Args:
            covars:     covariances to be considered in the sum
        """
        Covariance.__init__(self)
        self.dim = None
        self.covars = []
        for covar in covars:
            self.addCovariance(covar)
            covar.register(self.clear_all)

    def clear_all(self):
        self.clear_cache('default')

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
        cs = filter(lambda c: c.getNumberParams() > 0, self.covars)
        for c in cs:
            n = c.getNumberParams()
            istop = istart + n
            c.setParams(params[istart:istop])
            istart = istop
        self._notify()

    def getParams(self):
        istart = 0
        params = SP.zeros(self.getNumberParams())
        cs = filter(lambda c: c.getNumberParams() > 0, self.covars)
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

    #####################
    # Cached
    #####################
    @cached
    def K(self):
        K = SP.zeros((self.dim,self.dim))
        for i in range(len(self.covars)):
            K += self.getCovariance(i).K()
        return K

    @cached
    def Kcross(self):
        R = None
        for i in range(len(self.covars)):
            if not self.getCovariance(i).use_to_predict:    continue
            if R is None:
                R = self.covars[i].Kcross()
            else:
                _ = self.covars[i].Kcross()
                assert _.shape[0]==R.shape[0], 'Dimension mismatch.'
                assert _.shape[1]==R.shape[1], 'Dimension mismatch.'
                R += _
        return R

    @cached
    def K_grad_i(self,i):
        istart = 0
        for j in range(len(self.covars)):
            istop = istart + self.getCovariance(j).getNumberParams()
            if (i < istop):
                idx = i - istart
                return self.getCovariance(j).K_grad_i(idx)
            istart = istop
        return None

    def _calcNumberParams(self):
        self.n_params = 0
        for i in range(len(self.covars)):
            self.n_params += self.getCovariance(i).getNumberParams()
        return self.n_params

    ####################
    # Interpretable Params
    ####################
    def getInterParams(self):
        istart = 0
        interParams = SP.zeros(self.getNumberParams())
        for i in range(len(self.covars)):
            istop = istart + self.getCovariance(i).getNumberParams()
            params[istart:istop] = self.getCovariance(i).getInterParams()
            istart = istop
        return params

    def K_grad_interParam_i(self,i):
        istart = 0
        for j in range(len(self.covars)):
            istop = istart + self.getCovariance(j).getNumberParams()
            if (i < istop):
                idx = i - istart
                return self.getCovariance(j).K_grad_interParam_i(idx)
            istart = istop
        return None

    def setFIinv(self, value):
        self._FIinv = value
        istart = 0
        for i in range(len(self.covars)):
            istop = istart + self.getCovariance(i).getNumberParams()
            self.getCovariance(i).setFIinv(value[istart:istop][:,istart:istop])
            istart = istop
