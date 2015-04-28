from covar_base import Covariance
import pdb
import scipy as SP
from limix.core.type.cached import Cached, cached
import warnings

class SumCov(Covariance):

    def __init__(self,*covars):
        self.dim = None
        self.covars = []
        for covar in covars:
            self.addCovariance(covar)
        self.clear_all()

    #####################
    # Covars handling
    #####################
    def addCovariance(self,covar):
        if self.dim is None:
            self.dim = covar.dim
        else:
            assert covar.dim==self.dim, 'Dimension mismatch'
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
        for i in range(len(self.covars)):
            istop = istart + self.getCovariance(i).getNumberParams()
            self.getCovariance(i).setParams(params[istart:istop])
            istart = istop
        self.clear_all()

    def getParams(self):
        istart = 0
        params = SP.zeros(self.getNumberParams())
        for i in range(len(self.covars)):
            istop = istart + self.getCovariance(i).getNumberParams()
            params[istart:istop] = self.getCovariance(i).getParams()
            istart = istop
        return params

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
        warnings.warn('Method not available for combinator covariances. Set use_to_predict for single covariance terms.')

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
                assert _.shape[0]==R.shape[0], 'Dimension mismatch'
                assert _.shape[1]==R.shape[1], 'Dimension mismatch'
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
