import sys
sys.path.insert(0,'./../../..')
from limix.core.cobj import *
from limix.core.utils.eigen import *
import scipy as SP
import scipy.linalg as LA
import scipy.sparse.linalg as SLA
import scipy.stats.mstats as MST
import warnings
from covariance import Covariance

import pdb

class cov3kronSum(Covariance):

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
        self.P = C1.P
        self.setR(R1,R2)
        self._calcNumberParams()
        self._initParams()

    @cached_idxs
    def Cstar(self,i,j):
        return SP.dot(self.C[j].USi2().T,SP.dot(self.C[i].K(),self.C[j].USi2()))

    @cached_idxs
    def U_Cstar(self,i,j):
        S,RV = LA.eigh(self.Cstar(i,j))
        self.fill_cache_idxs('S_Cstar',S,(i,j))
        return RV

    @cached_idxs
    def S_Cstar(self,i,j):
        RV,U = LA.eigh(self.Cstar(i,j))
        self.fill_cache_idxs('U_Cstar',U,(i,j))
        return RV

    @cached_idxs
    def S_R(self,i):
        RV,U = LA.eigh(self.R[i])
        self.fill_cache_idxs('U_R',U,i)
        return RV

    @cached_idxs
    def U_R(self,i):
        S,RV = LA.eigh(self.R[i])
        self.fill_cache_idxs('S_R',S,i)
        return RV

    @cached_idxs
    def USi2_R(self,i):
        return self.U_R(i)*(self.S_R(i)**(-0.5))

    @cached_idxs
    def Rstar(self,i,j):
        if i==2:    RV = SP.diag(1./self.S_R(j))
        elif j==2:  RV = self.R[i]
        else:       RV = SP.dot(self.USi2_R(j).T,SP.dot(self.R[i],self.USi2_R(j)))
        return RV

    @cached_idxs
    def S_Rstar(self,i,j):
        if i==2:    RV = 1./self.S_R(j)
        if j==2:    RV = self.S_R(i)
        else:       RV = LA.eigh(self.Rstar(i,j),eigvals_only=True)
        return RV

    def CstarGrad(self,i,j,k,p):
        """
        Derivarive of Cstar_i_j with respect to the p-th param of cov k
        """
        if k==i:
            RV = SP.dot(self.C[j].USi2().T,SP.dot(self.C[i].Kgrad_param(p),self.C[j].USi2()))
        elif k==j:
            RV = SP.dot(self.C[j].USi2grad(p).T,SP.dot(self.C[i].K(),self.C[j].USi2()))
            RV+= SP.dot(self.C[j].USi2().T,SP.dot(self.C[i].K(),self.C[j].USi2grad(p)))
        else:
            RV = None
        return RV

    def S_CstarGrad(self,i,j,k,p):
        return dS_dti(self.CstarGrad(i,j,k,p),U=self.U_Cstar(i,j))

    def U_CstarGrad(self,i,j,k,p):
        return dU_dti(self.CstarGrad(i,j,k,p),U=self.U_Cstar(i,j),S=self.S_Cstar(i,j))

    def setR(self,R1,R2):
        self.N = R1.shape[0]
        self.R = [R1,R2]
        for i in range(3):
            self.clear_cache_idxs('S_R',i)
            self.clear_cache_idxs('U_R',i)
            self.clear_cache_idxs('USi2_R',i)
        for i in range(3):
            for j in range(3):
                self.clear_cache_idxs('Rstar',(i,j))
                self.clear_cache_idxs('S_Rstar',(i,j))

    def setParams(self,params):
        self.C[0].setParams(params[:self.C[0].getNumberParams()])
        self.C[1].setParams(params[self.C[0].getNumberParams():self.C[0].getNumberParams()+self.C[1].getNumberParams()])
        self.C[2].setParams(params[self.C[0].getNumberParams()+self.C[1].getNumberParams():])
        for i in range(3):
            for j in range(3):
                self.clear_cache_idxs('Cstar',(i,j))
                self.clear_cache_idxs('U_Cstar',(i,j))
                self.clear_cache_idxs('S_Cstar',(i,j))
        self.clear_cache('logdet_bound_1')
        self.clear_cache('logdet_bound_2')
        self.clear_cache('logdet_bound_3')

    def getParams(self):
        return SP.concatenate([self.C[0].getParams(),self.C[1].getParams(),self.C[2].getParams()])

    def _calcNumberParams(self):
        self.n_params = self.C[0].getNumberParams()+self.C[1].getNumberParams()+self.C[2].getNumberParams()

    def K(self):
        assert self.P*self.N<5000, 'dimension is too high'
        RV = SP.kron(self.C[0].K(),self.R[0])
        RV+= SP.kron(self.C[1].K(),self.R[1])
        RV+= SP.kron(self.C[2].K(),SP.eye(self.N))
        return RV

    def logdet(self):
        S,U = LA.eigh(self.K())
        return SP.log(S).sum()

    def _Kx(self,x):
        """
        compute K x
        """
        X = SP.reshape(x,(self.N,self.P),order='F')
        B = SP.dot(self.R[0],SP.dot(X,self.C[0].K()))
        B+= SP.dot(self.R[1],SP.dot(X,self.C[1].K()))
        B+= SP.dot(X,self.C[2].K())
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
    def logdet_bound_1(self):
        S2 = SP.kron(self.S_Cstar(1,0),self.S_Rstar(1,0))
        S3 = SP.kron(self.S_Cstar(2,0),self.S_Rstar(2,0))
        idx2 = S2.argsort()[::-1]
        idx3 = S3.argsort()
        RV = SP.log(S2[idx2]+S3[idx3]+1).sum()
        RV+= SP.sum(SP.log(self.C[0].S()))*self.N
        RV+= SP.sum(SP.log(self.S_R(0)))*self.P
        return RV

    def logdet_bound_1_grad_C1(self,p):
        S2 = SP.kron(self.S_Cstar(1,0),self.S_Rstar(1,0))
        S3 = SP.kron(self.S_Cstar(2,0),self.S_Rstar(2,0))
        S2grad = SP.kron(self.S_CstarGrad(1,0,0,p),self.S_Rstar(1,0))
        S3grad = SP.kron(self.S_CstarGrad(2,0,0,p),self.S_Rstar(2,0))
        idx2 = S2.argsort()[::-1]
        idx3 = S3.argsort()
        RV = ((S2grad[idx2]+S3grad[idx3])/(S2[idx2]+S3[idx3]+1)).sum()
        RV+= SP.sum(self.C[0].Sgrad(p)/self.C[0].S())*self.N
        return RV

    def logdet_bound_1_grad_C2(self,p):
        S2 = SP.kron(self.S_Cstar(1,0),self.S_Rstar(1,0))
        S3 = SP.kron(self.S_Cstar(2,0),self.S_Rstar(2,0))
        S2grad = SP.kron(self.S_CstarGrad(1,0,1,p),self.S_Rstar(1,0))
        idx2 = S2.argsort()[::-1]
        idx3 = S3.argsort()
        RV = (S2grad[idx2]/(S2[idx2]+S3[idx3]+1)).sum()
        return RV

    def logdet_bound_1_grad_C3(self,p):
        S2 = SP.kron(self.S_Cstar(1,0),self.S_Rstar(1,0))
        S3 = SP.kron(self.S_Cstar(2,0),self.S_Rstar(2,0))
        S3grad = SP.kron(self.S_CstarGrad(2,0,2,p),self.S_Rstar(2,0))
        idx2 = S2.argsort()[::-1]
        idx3 = S3.argsort()
        RV = (S3grad[idx3]/(S2[idx2]+S3[idx3]+1)).sum()
        return RV

    @cached
    def logdet_bound_2(self):
        S1 = SP.kron(self.S_Cstar(0,1),self.S_Rstar(0,1))
        S3 = SP.kron(self.S_Cstar(2,1),self.S_Rstar(2,1))
        idx1 = S1.argsort()[::-1]
        idx3 = S3.argsort()
        RV = SP.log(S1[idx1]+S3[idx3]+1).sum()
        RV+= SP.sum(SP.log(self.C[1].S()))*self.N
        RV+= SP.sum(SP.log(self.S_R(1)))*self.P
        return RV

    def logdet_bound_2_grad_C1(self,p):
        S1 = SP.kron(self.S_Cstar(0,1),self.S_Rstar(0,1))
        S3 = SP.kron(self.S_Cstar(2,1),self.S_Rstar(2,1))
        S1grad = SP.kron(self.S_CstarGrad(0,1,0,p),self.S_Rstar(0,1))
        idx1 = S1.argsort()[::-1]
        idx3 = S3.argsort()
        RV = (S1grad[idx1]/(S1[idx1]+S3[idx3]+1)).sum()
        return RV

    def logdet_bound_2_grad_C2(self,p):
        S1 = SP.kron(self.S_Cstar(0,1),self.S_Rstar(0,1))
        S3 = SP.kron(self.S_Cstar(2,1),self.S_Rstar(2,1))
        S1grad = SP.kron(self.S_CstarGrad(0,1,1,p),self.S_Rstar(0,1))
        S3grad = SP.kron(self.S_CstarGrad(2,1,1,p),self.S_Rstar(2,1))
        idx1 = S1.argsort()[::-1]
        idx3 = S3.argsort()
        RV = ((S1grad[idx1]+S3grad[idx3])/(S1[idx1]+S3[idx3]+1)).sum()
        RV+= SP.sum(self.C[1].Sgrad(p)/self.C[1].S())*self.N
        return RV

    def logdet_bound_2_grad_C3(self,p):
        S1 = SP.kron(self.S_Cstar(0,1),self.S_Rstar(0,1))
        S3 = SP.kron(self.S_Cstar(2,1),self.S_Rstar(2,1))
        S3grad = SP.kron(self.S_CstarGrad(2,1,2,p),self.S_Rstar(2,1))
        idx1 = S1.argsort()[::-1]
        idx3 = S3.argsort()
        RV = (S3grad[idx3]/(S1[idx1]+S3[idx3]+1)).sum()
        return RV

    @cached
    def logdet_bound_3(self):
        S1 = SP.kron(self.S_Cstar(0,2),self.S_Rstar(0,2))
        S2 = SP.kron(self.S_Cstar(1,2),self.S_Rstar(1,2))
        idx1 = S1.argsort()[::-1]
        idx2 = S2.argsort()
        RV = SP.log(S1[idx1]+S2[idx2]+1).sum()
        RV+= SP.sum(SP.log(self.C[2].S()))*self.N
        return RV

    def logdet_bound_3_grad_C1(self,p):
        S1 = SP.kron(self.S_Cstar(0,2),self.S_Rstar(0,2))
        S2 = SP.kron(self.S_Cstar(1,2),self.S_Rstar(1,2))
        S1grad = SP.kron(self.S_CstarGrad(0,2,0,p),self.S_Rstar(0,2))
        idx1 = S1.argsort()[::-1]
        idx2 = S2.argsort()
        RV = (S1grad[idx1]/(S1[idx1]+S2[idx2]+1)).sum()
        return RV

    def logdet_bound_3_grad_C2(self,p):
        S1 = SP.kron(self.S_Cstar(0,2),self.S_Rstar(0,2))
        S2 = SP.kron(self.S_Cstar(1,2),self.S_Rstar(1,2))
        S2grad = SP.kron(self.S_CstarGrad(1,2,1,p),self.S_Rstar(1,2))
        idx1 = S1.argsort()[::-1]
        idx2 = S2.argsort()
        RV = (S2grad[idx2]/(S1[idx1]+S2[idx2]+1)).sum()
        return RV

    def logdet_bound_3_grad_C3(self,p):
        S1 = SP.kron(self.S_Cstar(0,2),self.S_Rstar(0,2))
        S2 = SP.kron(self.S_Cstar(1,2),self.S_Rstar(1,2))
        S1grad = SP.kron(self.S_CstarGrad(0,2,2,p),self.S_Rstar(0,2))
        S2grad = SP.kron(self.S_CstarGrad(1,2,2,p),self.S_Rstar(1,2))
        idx1 = S1.argsort()[::-1]
        idx2 = S2.argsort()
        RV = ((S1grad[idx1]+S2grad[idx2])/(S1[idx1]+S2[idx2]+1)).sum()
        RV+= SP.sum(self.C[2].Sgrad(p)/self.C[2].S())*self.N
        return RV

    def logdet_bound(self,bound=2):
        if bound==0:      RV = self.logdet_bound_1()
        elif bound==1:    RV = self.logdet_bound_2()
        elif bound==2:    RV = self.logdet_bound_3()
        return RV

    def logdet_bound_grad_C1(self,p,bound=2):
        if bound==0:      RV = self.logdet_bound_1_grad_C1(p)
        elif bound==1:    RV = self.logdet_bound_2_grad_C1(p)
        elif bound==2:    RV = self.logdet_bound_3_grad_C1(p)
        return RV

    def logdet_bound_grad_C2(self,p,bound=2):
        if bound==0:      RV = self.logdet_bound_1_grad_C2(p)
        elif bound==1:    RV = self.logdet_bound_2_grad_C2(p)
        elif bound==2:    RV = self.logdet_bound_3_grad_C2(p)
        return RV

    def logdet_bound_grad_C3(self,p,bound=2):
        if bound==0:      RV = self.logdet_bound_1_grad_C3(p)
        elif bound==1:    RV = self.logdet_bound_2_grad_C3(p)
        elif bound==2:    RV = self.logdet_bound_3_grad_C3(p)
        return RV

    def numGrad_C1(self,f,h=1e-6):
        params1  = self.C[0].getParams().copy()
        params2  = self.C[1].getParams().copy()
        params3  = self.C[2].getParams().copy()
        params    = params1.copy()
        RV = []
        for i in range(params.shape[0]):
            params[i] = params1[i]+h
            _params = SP.concatenate([params,params2,params3])
            self.setParams(_params)
            fR = f()
            params[i] = params1[i]-h
            _params = SP.concatenate([params,params2,params3])
            self.setParams(_params)
            fL = f()
            params[i] = params1[i]
            RV.append((fR-fL)/(2*h))
        return SP.array(RV)

    def numGrad_C2(self,f,h=1e-6):
        params1  = self.C[0].getParams().copy()
        params2  = self.C[1].getParams().copy()
        params3  = self.C[2].getParams().copy()
        params    = params2.copy()
        RV = []
        for i in range(params.shape[0]):
            params[i] = params2[i]+h
            _params = SP.concatenate([params1,params,params3])
            self.setParams(_params)
            fR = f()
            params[i] = params2[i]-h
            _params = SP.concatenate([params1,params,params3])
            self.setParams(_params)
            fL = f()
            params[i] = params2[i]
            RV.append((fR-fL)/(2*h))
        return SP.array(RV)

    def numGrad_C3(self,f,h=1e-6):
        params1  = self.C[0].getParams().copy()
        params2  = self.C[1].getParams().copy()
        params3  = self.C[2].getParams().copy()
        params    = params3.copy()
        RV = []
        for i in range(params.shape[0]):
            params[i] = params3[i]+h
            _params = SP.concatenate([params1,params2,params])
            self.setParams(_params)
            fR = f()
            params[i] = params3[i]-h
            _params = SP.concatenate([params1,params2,params])
            self.setParams(_params)
            fL = f()
            params[i] = params3[i]
            RV.append((fR-fL)/(2*h))
        return SP.array(RV)
