import sys
sys.path.insert(0,'./../../..')
from limix.core.utils.observed import Observed
from limix.core.utils.cached import *
from limix.utils.preprocess import regressOut
import scipy as sp
import numpy as np
		    
import scipy.linalg as LA
import copy
import pdb

class mean_base(cObject, Observed):

    def __init__(self,Y,F):
        """
        y:        phenotype vector
        F:        fixed effect design
        """
        self.Y = Y
        self.F = F
        self.B = sp.zeros((self._K,1))

    #########################################
    # Properties 
    #########################################
    @property
    def Y(self):
        return self._Y

    @property
    def F(self):
        return self._F

    @property
    def B(self):
        return self._B

    @property
    def y(self):
        return sp.reshape(self.Y,(self._N*self._P,1),order='F') 

    @property
    def b(self):
        return sp.reshape(self.B,(self._K*self._P,1),order='F')

    #########################################
    # Setters 
    #########################################
    @Y.setter
    def Y(self,value):
        """ set phenotype """
        self._N = value.shape[0]
        self._P = value.shape[1]
        self._Y = value
        self.clear_cache('Yres')

    @F.setter
    def F(self,value):
        """ set phenotype """
        assert value.shape[0]==self._N, 'Dimension mismatch'
        self._K = value.shape[1]
        self._F = value
        self.clear_cache('predict','Yres')

    @B.setter
    def B(self,value):
        """ set phenotype """
        assert value.shape[0]==self._K, 'Dimension mismatch'
        assert value.shape[1]==self._P, 'Dimension mismatch'
        self._B = value
        self.clear_cache('predict','Yres')

    #########################################
    # Predictions 
    #########################################
    def predict(self,Fstar=None):
        if Fstar is None:
            r = self._predict_in_sample()
        else:
            r = self._predict_out_sample()
        return r 

    @cached
    def _predict_in_sample(self):
        r = _predict_fun(self.F) 

    def _predict_out_sample(self,Fstar):
        assert Fstar.shape[1]==self._K, 'Dimension mismatch'
        r = _predict_fun(Fstar) 

    def _predict_fun(self,M):
        return sp.dot(M,self.B)

    @cached
    def Yres(self):
        """ residual """
        RV  = self.Y-self.predict() 
        return RV

    ###########################################
    # Gradient TODO
    ###########################################
    #def getGradient(self,j):
    #    """ get gradient for fixed effect i """
    #    return rv

    #########################################
    # Params manipulation TODO 
    #########################################
    #def getParams(self):
    #    """ get params """
    #    return rv

    #def setParams(self,params):
    #    """ set params """
    #    start = 0
    #    for i in range(self.n_terms):
    #        n_effects = self.B[i].size
    #        self.B[i] = np.reshape(params[start:start+n_effects],self.B[i].shape, order='F')
    #        start += n_effects

