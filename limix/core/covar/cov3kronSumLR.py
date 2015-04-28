import sys
sys.path.insert(0,'./../../..')
from limix.core.cobj import *
from limix.core.utils.eigen import *
from limix.core.covar import cov2kronSum
import scipy as SP
import scipy.linalg as LA
import scipy.sparse.linalg as SLA
import scipy.stats.mstats as MST
import warnings
from covariance import Covariance

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
        self.clear_cache('Cstar','S_Cstar','U_Cstar','S','D','Lc','CrStar','S_CrStar','logdet','logdet_up','logdet_low','logdet_rnd')

    def getParams(self):
        return SP.concatenate([Cr.getParams(),Cg.getParams(),Cn.getParams()])

    def _calcNumberParams(self):
        self.n_params = self.Cr.getNumberParams()+self.Cg.getNumberParams()+self.Cn.getNumberParams()

    def setXX(self,XX):
        self.XX = XX
        self.N  = XX.shape[0]
        self.clear_cache('Sr','Lr','GGstar','S_GGstar','U_GGstar','S','D','logdet','logdet_up','logdet_low','logdet_rnd')

    def setGG(self,GG):
        self.GG = GG
        self.clear_cache('GGstar','S_GGstar','U_GGstar','logdet','logdet_up','logdet_low')

    def K(self):
        assert self.XX.shape[0]*self.Cr.K().shape[0]<5000, 'dimension is too high'
        RV = SP.kron(self.Cr.K(),self.GG)
        RV+= SP.kron(self.Cg.K(),self.XX)
        RV+= SP.kron(self.Cn.K(),SP.eye(self.N))
        return RV

    def _Kx(self,x):
        """
        compute K x
        """
        X = SP.reshape(x,(self.N,self.P),order='F')
        B = SP.dot(self.GG,SP.dot(X,self.Cr.K()))
        B+= SP.dot(self.XX,SP.dot(X,self.Cg.K()))
        B+= SP.dot(X,self.Cn.K())
        b = SP.reshape(B,(self.N*self.P),order='F')
        return b

    def solve(self,Y,X0=None,tol=1E-9):
        """ solve lin system Ki Y = X """
        Kx_linop = SLA.LinearOperator((self.N*self.P,self.N*self.P),matvec=self._Kx,rmatvec=self._Kx,dtype='float64')
        x0 = SP.reshape(X0,(self.N*self.P),order='F')
        y  = SP.reshape(Y,(self.N*self.P),order='F')
        x,_ = SLA.cgs(Kx_linop,y,x0=x0,tol=tol)
        RV = SP.reshape(x,(self.N,self.P),order='F')
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
        self.fill_cache('U_CrStar',U)
        return S

    @cached
    def U_CrStar(self):
        S,U = LA.eigh(self.CrStar())
        self.fill_cache('S_CrStar',S)
        return U

    @cached
    def GGstar(self):
        return SP.dot(self.Lr(),SP.dot(self.GG,self.Lr().T))

    @cached
    def S_GGstar(self):
        S,U = LA.eigh(self.GGstar())
        self.fill_cache('U_GGstar',U)
        return S

    @cached
    def U_GGstar(self):
        S,U = LA.eigh(self.GGstar())
        self.fill_cache('S_GGstar',S)
        return U

    def logdet_bound(self,bound='up'):
        Sr = SP.kron(self.S_CrStar(),self.S_GGstar())
        if bound=='up':
            idx_r = Sr.argsort()[::-1]
        elif bound=='low':
            idx_r = Sr.argsort()
        idx = self.S().argsort()
        RV = SP.log(Sr[idx_r]+self.S()[idx]).sum()
        RV+= SP.sum(SP.log(self.Cn.S()))*self.N
        return RV

    def logdet_rnd(self,n_perms=1):
        Sr = SP.kron(self.S_CrStar(),self.S_GGstar())
        RV = []
        for n in range(n_perms):
            idx_r = SP.random.permutation(self.N*self.P)
            RV.append(SP.log(Sr[idx_r]+self.S()).sum())
        RV = MST.gmean(RV)
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

    def logdet_low_debug(self):
        assert self.XX.shape[0]*self.Cr.K().shape[0]<5000, 'dimension is too high'
        A = SP.kron(self.Cr.K(),self.GG)
        B = SP.kron(self.Cg.K(),self.XX)
        B+= SP.kron(self.Cn.K(),SP.eye(self.XX.shape[0]))
        Sa,Ua = LA.eigh(A)
        Sb,Ub = LA.eigh(B)
        return SP.log(Sa+Sb).sum()

    def CrStarGrad_r(self,i):
        return SP.dot(self.Lc(),SP.dot(self.Cr.Kgrad_param(i),self.Lc().T))

    def CrStarGrad_g(self,i):
        RV = SP.dot(self.LcGrad_g(i),SP.dot(self.Cr.K(),self.Lc().T))
        RV+= RV.T 
        return RV
        
    def CrStarGrad_n(self,i):
        RV = SP.dot(self.LcGrad_n(i),SP.dot(self.Cr.K(),self.Lc().T))
        RV+= RV.T 
        return RV

    def S_CrStarGrad_r(self,i):
        return dS_dti(self.CrStarGrad_r(i),U=self.U_CrStar())

    def S_CrStarGrad_g(self,i):
        return dS_dti(self.CrStarGrad_g(i),U=self.U_CrStar())

    def S_CrStarGrad_n(self,i):
        return dS_dti(self.CrStarGrad_n(i),U=self.U_CrStar())

    def logdet_bound_grad_r(self,i,bound='up'):
        Sr = SP.kron(self.S_CrStar(),self.S_GGstar())
        SrGrad = SP.kron(self.S_CrStarGrad_r(i),self.S_GGstar())
        if bound=='up':
            idx_r = Sr.argsort()[::-1]
        elif bound=='low':
            idx_r = Sr.argsort()
        idx = self.S().argsort()
        RV = (SrGrad[idx_r]/(Sr[idx_r]+self.S()[idx])).sum()
        return RV

    def logdet_bound_grad_g(self,i,bound='up'):
        Sr = SP.kron(self.S_CrStar(),self.S_GGstar())
        SrGrad = SP.kron(self.S_CrStarGrad_g(i),self.S_GGstar())
        Sgrad  = self.Sgrad_g(i)
        if bound=='up':
            idx_r = Sr.argsort()[::-1]
        elif bound=='low':
            idx_r = Sr.argsort()
        idx = self.S().argsort()
        RV = ((SrGrad[idx_r]+Sgrad[idx])/(Sr[idx_r]+self.S()[idx])).sum()
        return RV

    def logdet_bound_grad_n(self,i,bound='up'):
        Sr = SP.kron(self.S_CrStar(),self.S_GGstar())
        SrGrad = SP.kron(self.S_CrStarGrad_n(i),self.S_GGstar())
        Sgrad  = self.Sgrad_n(i)
        if bound=='up':
            idx_r = Sr.argsort()[::-1]
        elif bound=='low':
            idx_r = Sr.argsort()
        idx = self.S().argsort()
        RV = ((SrGrad[idx_r]+Sgrad[idx])/(Sr[idx_r]+self.S()[idx])).sum()
        RV+= SP.sum(self.Cn.Sgrad(i)/self.Cn.S())*self.N
        return RV

    def numGrad_r(self,f,h=1e-4):
        params_r  = self.Cr.getParams().copy()
        params_g  = self.Cg.getParams().copy()
        params_n  = self.Cn.getParams().copy()
        params    = params_r.copy()
        RV = []
        for i in range(params.shape[0]):
            params[i] = params_r[i]+h 
            _params = SP.concatenate([params,params_g,params_n])
            self.setParams(_params)
            fR = f()
            params[i] = params_r[i]-h 
            _params = SP.concatenate([params,params_g,params_n])
            self.setParams(_params)
            fL = f()
            params[i] = params_r[i]
            RV.append((fR-fL)/(2*h))
        return SP.array(RV)

    def numGrad_g(self,f,h=1e-4):
        params_r  = self.Cr.getParams().copy()
        params_g  = self.Cg.getParams().copy()
        params_n  = self.Cn.getParams().copy()
        params    = params_g.copy()
        RV = []
        for i in range(params.shape[0]):
            params[i] = params_g[i]+h 
            _params = SP.concatenate([params_r,params,params_n])
            self.setParams(_params)
            fR = f()
            params[i] = params_g[i]-h 
            _params = SP.concatenate([params_r,params,params_n])
            self.setParams(_params)
            fL = f()
            params[i] = params_g[i]
            RV.append((fR-fL)/(2*h))
        return SP.array(RV)

    def numGrad_n(self,f,h=1e-4):
        params_r  = self.Cr.getParams().copy()
        params_g  = self.Cg.getParams().copy()
        params_n  = self.Cn.getParams().copy()
        params    = params_n.copy()
        RV = []
        for i in range(params.shape[0]):
            params[i] = params_n[i]+h 
            _params = SP.concatenate([params_r,params_g,params])
            self.setParams(_params)
            fR = f()
            params[i] = params_n[i]-h 
            _params = SP.concatenate([params_r,params_g,params])
            self.setParams(_params)
            fL = f()
            params[i] = params_n[i]
            RV.append((fR-fL)/(2*h))
        return SP.array(RV)

