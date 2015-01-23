import sys
sys.path.insert(0,'./../../..')
from limix.core.cobj import *
from limix.core.utils.eigen import *
import scipy as SP
import scipy.linalg as LA
import warnings
from covariance import covariance

import pdb

# should be a child class of combinator
class cov2kronSum(covariance):

    def __init__(self,Cg=None,Cn=None,XX=None):
        """
        initialization
        """
        assert Cg is not None, 'specify Cg'
        assert Cn is not None, 'specify Cn'
        assert XX is not None, 'specify XX'
        self.Cg = Cg
        self.Cn = Cn
        self.P  = Cg.P
        self.setXX(XX)
        self._calcNumberParams()
        self._initParams()

    def setParams(self,params):
        self.Cg.setParams(params[:self.Cg.getNumberParams()])
        self.Cn.setParams(params[self.Cg.getNumberParams():])
        self.clear_cache('Cstar','S_Cstar','U_Cstar','S','D','Lc','logdet')

    def getParams(self):
        return SP.concatenate([self.Cg.getParams(),self.Cn.getParams()])

    def _calcNumberParams(self):
        self.n_params = self.Cg.getNumberParams()+self.Cn.getNumberParams()

    def setXX(self,XX):
        self.XX = XX
        self.N  = XX.shape[0]
        self.clear_cache('Sr','Lr','S','D','logdet')

    @cached
    def logdet(self):
        return SP.sum(SP.log(self.Cn.S()))*self.XX.shape[0] + SP.log(self.S()).sum() 

    @cached
    def Cstar(self):
        return SP.dot(self.Cn.USi2().T,SP.dot(self.Cg.K(),self.Cn.USi2()))

    @cached
    def S_Cstar(self):
        RV,U = LA.eigh(self.Cstar())
        self.fill_cache('U_Cstar',U) 
        return RV

    @cached
    def U_Cstar(self):
        S,RV = LA.eigh(self.Cstar())
        self.fill_cache('S_Cstar',S) 
        return RV

    @cached
    def S(self):
        return SP.kron(self.S_Cstar(),self.Sr())+1

    @cached
    def D(self):
        return 1./self.S()

    @cached
    def Lc(self):
        return SP.dot(self.U_Cstar().T,self.Cn.USi2().T) 

    @cached
    def Sr(self):
        RV,U = LA.eigh(self.XX)
        self.fill_cache('Lr',U.T)
        return RV

    @cached
    def Lr(self):
        S,U = LA.eigh(self.XX)
        self.fill_cache('Sr',S)
        return U.T

    def CstarGrad_g(self,i):
        return SP.dot(self.Cn.USi2().T,SP.dot(self.Cg.Kgrad_param(i),self.Cn.USi2()))

    def CstarGrad_n(self,i):
        RV = SP.dot(self.Cn.USi2grad(i).T,SP.dot(self.Cg.K(),self.Cn.USi2()))
        RV+= RV.T

    def S_CstarGrad_g(self,i):
        return dS_dti(self.CstarGrad_g(i),U=self.U_Cstar())

    def U_CstarGrad_g(self,i):
        return dU_dti(self.CstarGrad_g(i),U=self.U_Cstar(),S=self.S_Cstar())

    def S_CstarGrad_n(self,i):
        return dS_dti(self.CstarGrad_n(i),U=self.U_Cstar())

    def U_CstarGrad_n(self,i):
        return dU_dti(self.CstarGrad_n(i),U=self.U_Cstar(),S=self.S_Cstar())

    def LcGrad_g(self,i):
        return SP.dot(self.U_CstarGrad_g().T,self.Cn.USi2().T) 

    def LcGrad_n(self,i):
        RV  = SP.dot(self.U_CstarGrad_n().T,self.Cn.USi2().T) 
        RV += SP.dot(self.U_Cstar().T,self.Cn.USi2grad().T) 
        return RV

    def Sgrad_g(self):
        return SP.kron(self.S_CstarGrad_g(),self.Sr())

    def Sgrad_n(self):
        return SP.kron(self.S_CstarGrad_n(),self.Sr())

