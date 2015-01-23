import sys
sys.path.insert(0,'./../../..')
from limix.core.cobj import *
from limix.core.utils.eigen import *
from limix.core.covar import cov2kronSum
import scipy as SP
import scipy.linalg as LA
import warnings
from covariance import covariance

import pdb

# should be a child class of combinator
class cov3kronSum(cov2kronSum):

    def __init__(self,Cr=None,Cg=None,Cn=None,GG=None,XX=None):
        """
        initialization
        """
        assert Cr is not None, 'specify Cr'
        assert Cg is not None, 'specify Cg'
        assert Cn is not None, 'specify Cn'
        assert GG is not None, 'specify GG'
        assert XX is not None, 'specify XX'
        self.Cr = Cr
        self.Cg = Cg
        self.Cn = Cn
        self.P  = Cg.P
        self.setXX(XX)
        self.setGG(GG)
        self._calcNumberParams()
        self._initParams()

    def setParams(self,params):
        self.Cr.setParams(params[:self.Cr.getNumberParams()])
        self.Cg.setParams(params[self.Cr.getNumberParams():self.Cr.getNumberParams()+self.Cg.getNumberParams()])
        self.Cn.setParams(params[self.Cr.getNumberParams()+self.Cg.getNumberParams():])
        self.clear_cache('Cstar','S_Cstar','U_Cstar','S','D','Lc','CrStar','S_CrStar','logdet','logdet_up','logdet_down')

    def getParams(self):
        return SP.concatenate([Cr.getParams(),Cg.getParams(),Cn.getParams()])

    def _calcNumberParams(self):
        self.n_params = self.Cr.getNumberParams()+self.Cg.getNumberParams()+self.Cn.getNumberParams()

    def setXX(self,XX):
        self.XX = XX
        self.N  = XX.shape[0]
        self.clear_cache('Sr','Lr','GGstar','S_GGstar','S','D','logdet','logdet_up','logdet_down')

    def setGG(self,GG):
        self.GG = GG
        self.clear_cache('GGstar','S_GGstar','logdet','logdet_up','logdet_down')

    def K(self):
        assert self.XX.shape[0]*self.Cr.K().shape[0]<5000, 'dimension is too high'
        RV = SP.kron(self.Cr.K(),self.GG)
        RV+= SP.kron(self.Cg.K(),self.XX)
        RV+= SP.kron(self.Cn.K(),SP.eye(self.N))
        return RV

    def K_2terms(self):
        assert self.XX.shape[0]*self.Cr.K().shape[0]<5000, 'dimension is too high'
        RV = SP.kron(self.Cg.K(),self.XX)
        RV+= SP.kron(self.Cn.K(),SP.eye(self.N))
        return RV

    @cached
    def logdet(self):
        S,U = LA.eigh(self.K())
        return SP.log(S).sum()

    @cached
    def CrStar(self):
        return SP.dot(self.Lc(),SP.dot(self.Cr.K(),self.Lc().T))

    @cached
    def S_CrStar(self):
        S,U = LA.eigh(self.CrStar())
        return S

    @cached
    def GGstar(self):
        return SP.dot(self.Lr(),SP.dot(self.GG,self.Lr().T))

    @cached
    def S_GGstar(self):
        S,U = LA.eigh(self.GGstar())
        return S

    @cached
    def logdet_up(self):
        Sr = SP.kron(self.S_CrStar(),self.S_GGstar())
        idx = self.S().argsort()
        idx_r = Sr.argsort()[::-1]
        RV = SP.log(Sr[idx_r]+self.S()[idx]).sum()
        RV+= SP.sum(SP.log(self.Cn.S()))*self.N
        return RV

    @cached
    def logdet_down(self):
        Sr = SP.kron(self.S_CrStar(),self.S_GGstar())
        idx = self.S().argsort()
        idx_r = Sr.argsort()
        RV = SP.log(Sr[idx_r]+self.S()[idx]).sum()
        RV+= SP.sum(SP.log(self.Cn.S()))*self.N
        return RV
        
    def logdet_up_debug(self):
        assert self.XX.shape[0]*self.Cr.K().shape[0]<5000, 'dimension is too high'
        A = SP.kron(self.Cr.K(),self.GG)
        B = SP.kron(self.Cg.K(),self.XX)
        B+= SP.kron(self.Cn.K(),SP.eye(self.XX.shape[0]))
        Sa,Ua = LA.eigh(A)
        Sb,Ub = LA.eigh(B)
        return SP.log(Sa+Sb[::-1]).sum()

    def logdet_down_debug(self):
        assert self.XX.shape[0]*self.Cr.K().shape[0]<5000, 'dimension is too high'
        A = SP.kron(self.Cr.K(),self.GG)
        B = SP.kron(self.Cg.K(),self.XX)
        B+= SP.kron(self.Cn.K(),SP.eye(self.XX.shape[0]))
        Sa,Ua = LA.eigh(A)
        Sb,Ub = LA.eigh(B)
        return SP.log(Sa+Sb).sum()

