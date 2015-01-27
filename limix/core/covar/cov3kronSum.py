import sys
sys.path.insert(0,'./../../..')
from limix.core.cobj import *
from limix.core.utils.eigen import *
import scipy as SP
import scipy.linalg as LA
import scipy.sparse.linalg as SLA
import scipy.stats.mstats as MST
import warnings
from covariance import covariance

import pdb

class cov3kronSum(covariance):

    def __init__(self,C1=None,C2=None,C3=None,R1=None,R2=None):
        """
        initialization
        """
        assert C1 is not None, 'specify C1'
        assert C2 is not None, 'specify C2'
        assert C3 is not None, 'specify C3'
        assert R1 is not None, 'specify R1'
        assert R2 is not None, 'specify R2'
        self.C = [C1,C2,C3]
        self.P  = C1.P
        self.setR1(R1)
        self.setR2(R2)
        self._calcNumberParams()
        self._initParams()

    def Cstar(i,j):
        if self.cache['C%dstar%d'] is None:
            RV = SP.dot(self.C[j].USi2().T,SP.dot(self.C[i].K(),self.C[j].USi2()))
        else:
            return RV

    def U_Cstar(i,j):
        name = 'C%dstar%d'%(i,j)
        S,RV = LA.eigh(getattr(self,name)())
        self.fill_cache('S_'+name,S)
        return RV

    ##########################################
    # Pulling out the third covariance
    ##########################################

    @cached
    def C1star3(self):      return Cstar(1,3) 
    @cached
    def C2star3(self):      return Cstar(2,3)

    @cached
    def S_C1star3(self):
        RV,U = LA.eigh(self.C1star3())
        self.fill_cache('U_C1star3',U)
        return RV

    @cached
    def U_C1star3(self):
        S,RV = LA.eigh(self.C1star3())
        self.fill_cache('S_C1star3',S)
        return RV

    @cached
    def S_C2star3(self):
        RV,U = LA.eigh(self.C2star3())
        self.fill_cache('U_C2star3',U)
        return RV

    @cached
    def U_C2star3(self):
        S,RV = LA.eigh(self.C2star3())
        self.fill_cache('S_C2star3',S)
        return R

    def C1star3Grad_1(self,i):
        return SP.dot(self.C3.USi2().T,SP.dot(self.C1.Kgrad_param(i),self.C3.USi2()))

    def C1star3Grad_3(self,i):
        RV = SP.dot(self.C3.USi2grad(i).T,SP.dot(self.C1.K(),self.C3.USi2()))
        RV+= RV.T
        return RV

    def S_C1star3Grad_1(self,i):
        return dS_dti(self.Cstar3Grad_1(i),U=self.U_Cstar())

    def U_C1star3Grad_1(self,i):
        return dU_dti(self.C1star3Grad_1(i),U=self.U_Cstar(),S=self.S_Cstar())

    @cached
    def R1star3(self):
        return R1

    @cached
    def R2star3(self):
        return R2



    @cached
    def C1star2(self):
        return SP.dot(self.C2.USi2().T,SP.dot(self.C1.K(),self.C2.USi2()))

    @cached
    def C3star2(self):
        return SP.dot(self.C2.USi2().T,SP.dot(self.C3.K(),self.C2.USi2()))

    @cached
    def C2star1(self):
        return SP.dot(self.C1.USi2().T,SP.dot(self.C2.K(),self.C1.USi2()))

    @cached
    def C3star1(self):
        return SP.dot(self.C1.USi2().T,SP.dot(self.C3.K(),self.C1.USi2()))

    @cached
    def S_C1star3(self):
        RV,U = LA.eigh(self.C1star3())
        self.fill_cache('U_C1star3',U)
        return RV

    @cached
    def U_C1star3(self):
        S,RV = LA.eigh(self.C1star3())
        self.fill_cache('S_C1star3',S)
        return RV

    @cached
    def S_C2star3(self):
        RV,U = LA.eigh(self.C2star3())
        self.fill_cache('U_C2star3',U)
        return RV

    @cached
    def U_C2star3(self):
        S,RV = LA.eigh(self.C2star3())
        self.fill_cache('S_C2star3',S)
        return RV

    @cached
    def S_C1star3(self):
        RV,U = LA.eigh(self.C1star3())
        self.fill_cache('U_C1star3',U)
        return RV

    @cached
    def U_C1star3(self):
        S,RV = LA.eigh(self.C1star3())
        self.fill_cache('S_C1star3',S)
        return RV

    @cached
    def S_C2star3(self):
        RV,U = LA.eigh(self.C2star3())
        self.fill_cache('U_C2star3',U)
        return RV

    @cached
    def U_C2star3(self):
        S,RV = LA.eigh(self.C2star3())
        self.fill_cache('S_C2star3',S)
        return RV

    def setParams(self,params):
        self.C1.setParams(params[:self.C1.getNumberParams()])
        self.C2.setParams(params[self.C1.getNumberParams():self.C1.getNumberParams()+self.C2.getNumberParams()])
        self.C3.setParams(params[self.C1.getNumberParams()+self.C2.getNumberParams():])
        self.clear_cache('Cstar','S_Cstar','U_Cstar','S','D','Lc','C1Star','S_C1Star','logdet','logdet_up','logdet_low','logdet_rnd')

    def getParams(self):
        return SP.concatenate([C1.getParams(),C2.getParams(),C3.getParams()])

    def _calcNumberParams(self):
        self.n_params = self.C1.getNumberParams()+self.C2.getNumberParams()+self.C3.getNumberParams()

    def setR2(self,R2):
        self.R2 = R2
        self.N  = R2.shape[0]
        self.clear_cache('Sr','Lr','R1star','S_R1star','U_R1star','S','D','logdet','logdet_up','logdet_low','logdet_rnd')

    def setR1(self,R1):
        self.R1 = R1
        self.clear_cache('R1star','S_R1star','U_R1star','logdet','logdet_up','logdet_low')

    def K(self):
        assert self.R2.shape[0]*self.C1.K().shape[0]<5000, 'dimension is too high'
        RV = SP.kron(self.C1.K(),self.R1)
        RV+= SP.kron(self.C2.K(),self.R2)
        RV+= SP.kron(self.C3.K(),SP.eye(self.N))
        return RV

    def _Kx(self,x):
        """
        compute K x
        """
        X = SP.reshape(x,(self.N,self.P),order='F')
        B = SP.dot(self.R1,SP.dot(X,self.C1.K()))
        B+= SP.dot(self.R2,SP.dot(X,self.C2.K()))
        B+= SP.dot(X,self.C3.K())
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
    def C1_g(self):
        return SP.dot(self.Lc(),SP.dot(self.C1.K(),self.Lc().T))

    @cached
    def S_C1Star(self):
        S,U = LA.eigh(self.C1Star())
        self.fill_cache('U_C1Star',U)
        return S

    @cached
    def U_C1Star(self):
        S,U = LA.eigh(self.C1Star())
        self.fill_cache('S_C1Star',S)
        return U

    @cached
    def R1star(self):
        return SP.dot(self.Lr(),SP.dot(self.R1,self.Lr().T))

    @cached
    def S_R1star(self):
        S,U = LA.eigh(self.R1star())
        self.fill_cache('U_R1star',U)
        return S

    @cached
    def U_R1star(self):
        S,U = LA.eigh(self.R1star())
        self.fill_cache('S_R1star',S)
        return U

    def logdet_bound(self,bound='up'):
        Sr = SP.kron(self.S_C1Star(),self.S_R1star())
        if bound=='up':
            idx_r = Sr.argsort()[::-1]
        elif bound=='low':
            idx_r = Sr.argsort()
        idx = self.S().argsort()
        RV = SP.log(Sr[idx_r]+self.S()[idx]).sum()
        RV+= SP.sum(SP.log(self.C3.S()))*self.N
        return RV

    def logdet_rnd(self,n_perms=1):
        Sr = SP.kron(self.S_C1Star(),self.S_R1star())
        RV = []
        for n in range(n_perms):
            idx_r = SP.random.permutation(self.N*self.P)
            RV.append(SP.log(Sr[idx_r]+self.S()).sum())
        RV = MST.gmean(RV)
        RV+= SP.sum(SP.log(self.C3.S()))*self.N
        return RV
        
    def logdet_up_debug(self):
        assert self.R2.shape[0]*self.C1.K().shape[0]<5000, 'dimension is too high'
        A = SP.kron(self.C1.K(),self.R1)
        B = SP.kron(self.C2.K(),self.R2)
        B+= SP.kron(self.C3.K(),SP.eye(self.R2.shape[0]))
        Sa,Ua = LA.eigh(A)
        Sb,Ub = LA.eigh(B)
        return SP.log(Sa+Sb[::-1]).sum()

    def logdet_low_debug(self):
        assert self.R2.shape[0]*self.C1.K().shape[0]<5000, 'dimension is too high'
        A = SP.kron(self.C1.K(),self.R1)
        B = SP.kron(self.C2.K(),self.R2)
        B+= SP.kron(self.C3.K(),SP.eye(self.R2.shape[0]))
        Sa,Ua = LA.eigh(A)
        Sb,Ub = LA.eigh(B)
        return SP.log(Sa+Sb).sum()

    def C1StarGrad_r(self,i):
        return SP.dot(self.Lc(),SP.dot(self.C1.Kgrad_param(i),self.Lc().T))

    def C1StarGrad_g(self,i):
        RV = SP.dot(self.LcGrad_g(i),SP.dot(self.C1.K(),self.Lc().T))
        RV+= RV.T 
        return RV
        
    def C1StarGrad_n(self,i):
        RV = SP.dot(self.LcGrad_n(i),SP.dot(self.C1.K(),self.Lc().T))
        RV+= RV.T 
        return RV

    def S_C1StarGrad_r(self,i):
        return dS_dti(self.C1StarGrad_r(i),U=self.U_C1Star())

    def S_C1StarGrad_g(self,i):
        return dS_dti(self.C1StarGrad_g(i),U=self.U_C1Star())

    def S_C1StarGrad_n(self,i):
        return dS_dti(self.C1StarGrad_n(i),U=self.U_C1Star())

    def logdet_bound_grad_r(self,i,bound='up'):
        Sr = SP.kron(self.S_C1Star(),self.S_R1star())
        SrGrad = SP.kron(self.S_C1StarGrad_r(i),self.S_R1star())
        if bound=='up':
            idx_r = Sr.argsort()[::-1]
        elif bound=='low':
            idx_r = Sr.argsort()
        idx = self.S().argsort()
        RV = (SrGrad[idx_r]/(Sr[idx_r]+self.S()[idx])).sum()
        return RV

    def logdet_bound_grad_g(self,i,bound='up'):
        Sr = SP.kron(self.S_C1Star(),self.S_R1star())
        SrGrad = SP.kron(self.S_C1StarGrad_g(i),self.S_R1star())
        Sgrad  = self.Sgrad_g(i)
        if bound=='up':
            idx_r = Sr.argsort()[::-1]
        elif bound=='low':
            idx_r = Sr.argsort()
        idx = self.S().argsort()
        RV = ((SrGrad[idx_r]+Sgrad[idx])/(Sr[idx_r]+self.S()[idx])).sum()
        return RV

    def logdet_bound_grad_n(self,i,bound='up'):
        Sr = SP.kron(self.S_C1Star(),self.S_R1star())
        SrGrad = SP.kron(self.S_C1StarGrad_n(i),self.S_R1star())
        Sgrad  = self.Sgrad_n(i)
        if bound=='up':
            idx_r = Sr.argsort()[::-1]
        elif bound=='low':
            idx_r = Sr.argsort()
        idx = self.S().argsort()
        RV = ((SrGrad[idx_r]+Sgrad[idx])/(Sr[idx_r]+self.S()[idx])).sum()
        RV+= SP.sum(self.C3.Sgrad(i)/self.C3.S())*self.N
        return RV

    def numGrad_r(self,f,h=1e-4):
        params_r  = self.C1.getParams().copy()
        params_g  = self.C2.getParams().copy()
        params_n  = self.C3.getParams().copy()
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
        params_r  = self.C1.getParams().copy()
        params_g  = self.C2.getParams().copy()
        params_n  = self.C3.getParams().copy()
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
        params_r  = self.C1.getParams().copy()
        params_g  = self.C2.getParams().copy()
        params_n  = self.C3.getParams().copy()
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

