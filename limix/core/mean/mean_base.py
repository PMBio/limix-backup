import sys
sys.path.insert(0,'./../../..')
from limix.core.cobj import * 
from limix.utils.preprocess import regressOut
import scipy as sp
import numpy as np
		    
import scipy.linalg as LA
import copy
import pdb

class mean_base(cObject):

    def __init__(self,Y,F,Fstar):
        """
        y:        phenotype vector
        F:        fixed effect design
        """
        self.Y = Y
        self.F = F
        self.Fstar = Fstar

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

    #########################################
    # Setters 
    #########################################
    @Y.setter
    def Y(self,value):
        """ set phenotype """
        assert value.shape[1]==1, 'Dimension mismatch'
        self._N = value.shape[0]
        self._Y = value
        #self.clear_cache()

    @F.setter
    def F(self,value):
        """ set phenotype """
        assert value.shape[0]==self._N, 'Dimension mismatch'
        self._K = value.shape[1]
        self._Y = value
        #self.clear_cache()

    @B.setter
    def B(self,value):
        """ set phenotype """
        assert value.shape[0]==self._K, 'Dimension mismatch'
        assert value.shape[1]==1, 'Dimension mismatch'
        self._B = value
        #self.clear_cache()

    #########################################
    # Getters (caching)
    #########################################
    @cached
    def predict(self):
        """ predict the value of the fixed effect (F*B) """
        return sp.dot(self.F,self.B)

    @cache
    def Yres(self):
        """ predict the value of """
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

