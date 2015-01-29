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

from scipy.optimize import fmin_l_bfgs_b as optimize

import pdb

# should be a child class of combinator
class cov3kronSum(covariance):

    def __init__(self,C1=None,C2=None,Cn=None,R1=None,R2=None):
        """
        initialization
        """
        assert C1 is not None, 'specify C1'
        assert C2 is not None, 'specify C2'
        assert Cn is not None, 'specify Cn'
        assert R1 is not None, 'specify R1'
        assert R2 is not None, 'specify R2'
        self.C1 = C1
        self.C2 = C2
        self.Cn = Cn
        self.P  = C2.P
        self.N  = R1.shape[0]
        self.setR1(R1)
        self.setR2(R2)
        self.setAB(0,0)
        self._calcNumberParams()
        self._initParams()

    def setParams(self,params):
        self.C1.setParams(params[:self.C1.getNumberParams()])
        self.C2.setParams(params[self.C1.getNumberParams():self.C1.getNumberParams()+self.C2.getNumberParams()])
        self.Cn.setParams(params[self.C1.getNumberParams()+self.C2.getNumberParams():])
        self.clear_cache('C3','S_C3','U_C3','USi2_C3')
        self.clear_cache('C1star','S_C1star','U_C1star')
        self.clear_cache('C2star','S_C2star','U_C2star')
        self.clear_cache('logdet','logdet_bound')

    def setAB(self,a,b):
        self.a = a
        self.b = b
        self.clear_cache('C3','S_C3','U_C3','USi2_C3')
        self.clear_cache('C1star','S_C1star','U_C1star')
        self.clear_cache('C2star','S_C2star','U_C2star')
        self.clear_cache('logdet','logdet_bound')

    def setR1(self,R1):
        self.R1 = R1
        self.clear_cache('S_R1','U_R1')
        self.clear_cache('logdet','logdet_bound')

    def setR2(self,R2):
        self.R2 = R2
        self.clear_cache('S_R2','U_R2')
        self.clear_cache('logdet','logdet_bound')

    def getParams(self):
        return SP.concatenate([self.C1.getParams(),self.C2.getParams(),self.Cn.getParams()])

    def _calcNumberParams(self):
        self.n_params = self.C1.getNumberParams()+self.C2.getNumberParams()+self.Cn.getNumberParams()

    def K(self):
        assert self.R2.shape[0]*self.C1.K().shape[0]<5000, 'dimension is too high'
        RV = SP.kron(self.C1.K(),self.R1)
        RV+= SP.kron(self.C2.K(),self.R2)
        RV+= SP.kron(self.Cn.K(),SP.eye(self.N))
        return RV

    def _Kx(self,x):
        """
        compute K x
        """
        X = SP.reshape(x,(self.N,self.P),order='F')
        B = SP.dot(self.R1,SP.dot(X,self.C1.K()))
        B+= SP.dot(self.R2,SP.dot(X,self.C2.K()))
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


    ###########################
    # R1, R2
    ##########################

    @cached
    def S_R1(self):
        S,U = LA.eigh(self.R1)
        self.fill_cache('U_R1',U)
        return S

    @cached
    def U_R1(self):
        S,U = LA.eigh(self.R1)
        self.fill_cache('S_R1',S)
        return U

    @cached
    def S_R2(self):
        S,U = LA.eigh(self.R2)
        self.fill_cache('U_R2',U)
        return S

    @cached
    def U_R2(self):
        S,U = LA.eigh(self.R2)
        self.fill_cache('S_R2',S)
        return U


    ###########################
    # C3
    ##########################

    @cached
    def C3(self):
        return self.Cn.K()+self.a*self.C1.K()+self.b*self.C2.K() 

    @cached
    def S_C3(self):
        S,U = LA.eigh(self.C3())
        self.fill_cache('U_C3',U)
        return S

    @cached
    def U_C3(self):
        S,U = LA.eigh(self.C3())
        self.fill_cache('S_C3',S)
        return U

    @cached
    def USi2_C3(self):
        return self.U_C3()*self.S_C3()**(-0.5)

    def C3_grad_1(self,i):
        return self.a*self.C1.Kgrad_param(i) 

    def C3_grad_2(self,i):
        return self.b*self.C2.Kgrad_param(i) 

    def C3_grad_n(self,i):
        return self.Cn.Kgrad_param(i) 

    def C3_grad_a(self):
        return self.C1.K()

    def C3_grad_b(self):
        return self.C2.K()

    def S_C3_grad_1(self,i):
        return dS_dti(self.C3_grad_1(i),U=self.U_C3())

    def S_C3_grad_2(self,i):
        return dS_dti(self.C3_grad_2(i),U=self.U_C3())

    def S_C3_grad_n(self,i):
        return dS_dti(self.C3_grad_n(i),U=self.U_C3())

    def S_C3_grad_a(self):
        return dS_dti(self.C3_grad_a(),U=self.U_C3())

    def S_C3_grad_b(self):
        return dS_dti(self.C3_grad_b(),U=self.U_C3())

    def U_C3_grad_1(self,i):
        return dU_dti(self.C3_grad_1(i),U=self.U_C3(),S=self.S_C3())

    def U_C3_grad_2(self,i):
        return dU_dti(self.C3_grad_2(i),U=self.U_C3(),S=self.S_C3())

    def U_C3_grad_n(self,i):
        return dU_dti(self.C3_grad_n(i),U=self.U_C3(),S=self.S_C3())

    def U_C3_grad_a(self):
        return dU_dti(self.C3_grad_a(),U=self.U_C3(),S=self.S_C3())

    def U_C3_grad_b(self):
        return dU_dti(self.C3_grad_b(),U=self.U_C3(),S=self.S_C3())

    def USi2_C3_grad_1(self,i):
        Si2grad = -0.5*self.S_C3()**(-1.5)*self.S_C3_grad_1(i)
        return self.U_C3_grad_1(i)*(self.S_C3()**(-0.5)) + self.U_C3()*Si2grad

    def USi2_C3_grad_2(self,i):
        Si2grad = -0.5*self.S_C3()**(-1.5)*self.S_C3_grad_2(i)
        return self.U_C3_grad_2(i)*(self.S_C3()**(-0.5)) + self.U_C3()*Si2grad

    def USi2_C3_grad_n(self,i):
        Si2grad = -0.5*self.S_C3()**(-1.5)*self.S_C3_grad_n(i)
        return self.U_C3_grad_n(i)*(self.S_C3()**(-0.5)) + self.U_C3()*Si2grad

    def USi2_C3_grad_a(self):
        Si2grad = -0.5*self.S_C3()**(-1.5)*self.S_C3_grad_a()
        return self.U_C3_grad_a()*(self.S_C3()**(-0.5)) + self.U_C3()*Si2grad

    def USi2_C3_grad_b(self):
        Si2grad = -0.5*self.S_C3()**(-1.5)*self.S_C3_grad_b()
        return self.U_C3_grad_b()*(self.S_C3()**(-0.5)) + self.U_C3()*Si2grad

    ###########################
    # C1star
    ##########################

    @cached
    def C1star(self):
        return SP.dot(self.USi2_C3().T,SP.dot(self.C1.K(),self.USi2_C3()))

    @cached
    def S_C1star(self):
        S,U = LA.eigh(self.C1star())
        self.fill_cache('U_C1star',U)
        return S

    @cached
    def U_C1star(self):
        S,U = LA.eigh(self.C1star())
        self.fill_cache('S_C1star',S)
        return U

    def C1star_grad_1(self,i):
        RV = SP.dot(self.USi2_C3_grad_1(i).T,SP.dot(self.C1.K(),self.USi2_C3()))
        RV+= RV.T
        RV+= SP.dot(self.USi2_C3().T,SP.dot(self.C1.Kgrad_param(i),self.USi2_C3()))
        return RV

    def C1star_grad_2(self,i):
        RV = SP.dot(self.USi2_C3_grad_2(i).T,SP.dot(self.C1.K(),self.USi2_C3()))
        RV+= RV.T
        return RV

    def C1star_grad_n(self,i):
        RV = SP.dot(self.USi2_C3_grad_n(i).T,SP.dot(self.C1.K(),self.USi2_C3()))
        RV+= RV.T
        return RV

    def C1star_grad_a(self):
        RV = SP.dot(self.USi2_C3_grad_a().T,SP.dot(self.C1.K(),self.USi2_C3()))
        RV+= RV.T
        return RV

    def C1star_grad_b(self):
        RV = SP.dot(self.USi2_C3_grad_b().T,SP.dot(self.C1.K(),self.USi2_C3()))
        RV+= RV.T
        return RV

    def S_C1star_grad_1(self,i):
        return dS_dti(self.C1star_grad_1(i),U=self.U_C1star())

    def S_C1star_grad_2(self,i):
        return dS_dti(self.C1star_grad_2(i),U=self.U_C1star())

    def S_C1star_grad_n(self,i):
        return dS_dti(self.C1star_grad_n(i),U=self.U_C1star())

    def S_C1star_grad_a(self):
        return dS_dti(self.C1star_grad_a(),U=self.U_C1star())

    def S_C1star_grad_b(self):
        return dS_dti(self.C1star_grad_b(),U=self.U_C1star())

    ###########################
    # C2star
    ##########################

    @cached
    def C2star(self):
        return SP.dot(self.USi2_C3().T,SP.dot(self.C2.K(),self.USi2_C3()))

    @cached
    def S_C2star(self):
        S,U = LA.eigh(self.C2star())
        self.fill_cache('U_C2star',U)
        return S

    @cached
    def U_C2star(self):
        S,U = LA.eigh(self.C2star())
        self.fill_cache('S_C2star',S)
        return U

    def C2star_grad_1(self,i):
        RV = SP.dot(self.USi2_C3_grad_1(i).T,SP.dot(self.C2.K(),self.USi2_C3()))
        RV+= RV.T
        return RV

    def C2star_grad_2(self,i):
        RV = SP.dot(self.USi2_C3_grad_2(i).T,SP.dot(self.C2.K(),self.USi2_C3()))
        RV+= RV.T
        RV+= SP.dot(self.USi2_C3().T,SP.dot(self.C2.Kgrad_param(i),self.USi2_C3()))
        return RV

    def C2star_grad_n(self,i):
        RV = SP.dot(self.USi2_C3_grad_n(i).T,SP.dot(self.C2.K(),self.USi2_C3()))
        RV+= RV.T
        return RV

    def C2star_grad_a(self):
        RV = SP.dot(self.USi2_C3_grad_a().T,SP.dot(self.C2.K(),self.USi2_C3()))
        RV+= RV.T
        return RV

    def C2star_grad_b(self):
        RV = SP.dot(self.USi2_C3_grad_b().T,SP.dot(self.C2.K(),self.USi2_C3()))
        RV+= RV.T
        return RV

    def S_C2star_grad_1(self,i):
        return dS_dti(self.C2star_grad_1(i),U=self.U_C2star())

    def S_C2star_grad_2(self,i):
        return dS_dti(self.C2star_grad_2(i),U=self.U_C2star())

    def S_C2star_grad_n(self,i):
        return dS_dti(self.C2star_grad_n(i),U=self.U_C2star())

    def S_C2star_grad_a(self):
        return dS_dti(self.C2star_grad_a(),U=self.U_C2star())

    def S_C2star_grad_b(self):
        return dS_dti(self.C2star_grad_b(),U=self.U_C2star())

    ###########################
    # Bound
    ##########################

    @cached
    def logdet_bound(self):
        S1 = SP.kron(self.S_C1star(),self.S_R1()-self.a)
        S2 = SP.kron(self.S_C2star(),self.S_R2()-self.b)
        idx1 = SP.argsort(S1)
        idx2 = SP.argsort(S2)
        if (S1[idx1]+S2[idx2]+1).min()<1e-4:
            RV = SP.inf
        else:
            RV = SP.log(S1[idx1][::-1]+S2[idx2]+1).sum()
            RV+= SP.sum(SP.log(self.S_C3()))*self.N
        return RV

    def logdet_bound_grad_1(self,i):
        S1 = SP.kron(self.S_C1star(),self.S_R1()-self.a)
        S2 = SP.kron(self.S_C2star(),self.S_R2()-self.b)
        S1grad = SP.kron(self.S_C1star_grad_1(i),self.S_R1()-self.a)
        S2grad = SP.kron(self.S_C2star_grad_1(i),self.S_R2()-self.b)
        idx1 = SP.argsort(S1)
        idx2 = SP.argsort(S2)
        assert (S1[idx1]+S2[idx2]+1).min()>1e-4, 'invalid values of a and b'
        RV = ((S1grad[idx1][::-1]+S2grad[idx2])/(S1[idx1][::-1]+S2[idx2]+1)).sum()
        RV+= (self.S_C3_grad_1(i)/self.S_C3()).sum()*self.N
        return RV

    def logdet_bound_grad_2(self,i):
        S1 = SP.kron(self.S_C1star(),self.S_R1()-self.a)
        S2 = SP.kron(self.S_C2star(),self.S_R2()-self.b)
        S1grad = SP.kron(self.S_C1star_grad_2(i),self.S_R1()-self.a)
        S2grad = SP.kron(self.S_C2star_grad_2(i),self.S_R2()-self.b)
        idx1 = SP.argsort(S1)
        idx2 = SP.argsort(S2)
        assert (S1[idx1]+S2[idx2]+1).min()>1e-4, 'invalid values of a and b'
        RV = ((S1grad[idx1][::-1]+S2grad[idx2])/(S1[idx1][::-1]+S2[idx2]+1)).sum()
        RV+= (self.S_C3_grad_2(i)/self.S_C3()).sum()*self.N
        return RV

    def logdet_bound_grad_n(self,i):
        S1 = SP.kron(self.S_C1star(),self.S_R1()-self.a)
        S2 = SP.kron(self.S_C2star(),self.S_R2()-self.b)
        S1grad = SP.kron(self.S_C1star_grad_n(i),self.S_R1()-self.a)
        S2grad = SP.kron(self.S_C2star_grad_n(i),self.S_R2()-self.b)
        idx1 = SP.argsort(S1)
        idx2 = SP.argsort(S2)
        assert (S1[idx1]+S2[idx2]+1).min()>1e-4, 'invalid values of a and b'
        RV = ((S1grad[idx1][::-1]+S2grad[idx2])/(S1[idx1][::-1]+S2[idx2]+1)).sum()
        RV+= (self.S_C3_grad_n(i)/self.S_C3()).sum()*self.N
        return RV

    def logdet_bound_grad_a(self):
        S1 = SP.kron(self.S_C1star(),self.S_R1()-self.a)
        S2 = SP.kron(self.S_C2star(),self.S_R2()-self.b)
        S1grad = SP.kron(self.S_C1star_grad_a(),self.S_R1()-self.a)
        S1grad+= SP.kron(self.S_C1star(),-SP.ones(self.N))
        S2grad = SP.kron(self.S_C2star_grad_a(),self.S_R2()-self.b)
        idx1 = SP.argsort(S1)
        idx2 = SP.argsort(S2)
        assert (S1[idx1]+S2[idx2]+1).min()>1e-4, 'invalid values of a and b'
        RV = ((S1grad[idx1][::-1]+S2grad[idx2])/(S1[idx1][::-1]+S2[idx2]+1)).sum()
        RV+= (self.S_C3_grad_a()/self.S_C3()).sum()*self.N
        return RV

    def logdet_bound_grad_b(self):
        S1 = SP.kron(self.S_C1star(),self.S_R1()-self.a)
        S2 = SP.kron(self.S_C2star(),self.S_R2()-self.b)
        S1grad = SP.kron(self.S_C1star_grad_b(),self.S_R1()-self.a)
        S2grad = SP.kron(self.S_C2star_grad_b(),self.S_R2()-self.b)
        S2grad+= SP.kron(self.S_C2star(),-SP.ones(self.N))
        idx1 = SP.argsort(S1)
        idx2 = SP.argsort(S2)
        assert (S1[idx1]+S2[idx2]+1).min()>1e-4, 'invalid values of a and b'
        RV = ((S1grad[idx1][::-1]+S2grad[idx2])/(S1[idx1][::-1]+S2[idx2]+1)).sum()
        RV+= (self.S_C3_grad_b()/self.S_C3()).sum()*self.N
        return RV

    def optimizeAB(self,values=None,n=None):
        """ Minimize the upper bound with respect to a and b """
        if n is None:
            n = 40
        if values is None:
            values = SP.linspace(0,1,n)
        ld_min = SP.inf
        A = SP.zeros((n,n))
        for ai in range(n):
            for bi in range(n):
                self.setAB(values[ai],values[bi])
                value = self.logdet_bound() 
                A[ai,bi] = value
                if value<ld_min:
                    ld_min = value
                    a_best = values[ai]
                    b_best = values[bi]
        import pylab as PL
        PL.ion()
        A[A==SP.inf] = 0
        PL.imshow(A)
        PL.colorbar()
        self.setAB(a_best,b_best)

    def optimizeABgrad(self):
        def f(x):
            self.setAB(x[0],x[1])
            rv = self.logdet_bound()
            rvGrad = SP.array([self.logdet_bound_grad_a(),self.logdet_bound_grad_b()])
            return rv,rvGrad
        x0 = SP.zeros(2)
        f(x0) 
        bounds = [(0,1e-2), (0,1e-2)]
        x,fmin,info = optimize(f, x0=x0, bounds=bounds)
        print x
        pdb.set_trace()


    def numGrad_1(self,f,h=1e-6):
        params_1  = self.C1.getParams().copy()
        params_2  = self.C2.getParams().copy()
        params_n  = self.Cn.getParams().copy()
        params    = params_1.copy()
        RV = []
        for i in range(params.shape[0]):
            params[i] = params_1[i]+h 
            _params = SP.concatenate([params,params_2,params_n])
            self.setParams(_params)
            fR = f()
            params[i] = params_1[i]-h 
            _params = SP.concatenate([params,params_2,params_n])
            self.setParams(_params)
            fL = f()
            params[i] = params_1[i]
            RV.append((fR-fL)/(2*h))
        return SP.array(RV)

    def numGrad_2(self,f,h=1e-6):
        params_1  = self.C1.getParams().copy()
        params_2  = self.C2.getParams().copy()
        params_n  = self.Cn.getParams().copy()
        params    = params_2.copy()
        RV = []
        for i in range(params.shape[0]):
            params[i] = params_2[i]+h 
            _params = SP.concatenate([params_1,params,params_n])
            self.setParams(_params)
            fR = f()
            params[i] = params_2[i]-h 
            _params = SP.concatenate([params_1,params,params_n])
            self.setParams(_params)
            fL = f()
            params[i] = params_2[i]
            RV.append((fR-fL)/(2*h))
        return SP.array(RV)

    def numGrad_n(self,f,h=1e-6):
        params_1  = self.C1.getParams().copy()
        params_2  = self.C2.getParams().copy()
        params_n  = self.Cn.getParams().copy()
        params    = params_n.copy()
        RV = []
        for i in range(params.shape[0]):
            params[i] = params_n[i]+h 
            _params = SP.concatenate([params_1,params_2,params])
            self.setParams(_params)
            fR = f()
            params[i] = params_n[i]-h 
            _params = SP.concatenate([params_1,params_2,params])
            self.setParams(_params)
            fL = f()
            params[i] = params_n[i]
            RV.append((fR-fL)/(2*h))
        return SP.array(RV)

    def numGrad_a(self,f,h=1e-6):
        a = self.a
        self.setAB(a+h,self.b)
        fR = f()
        self.setAB(a-h,self.b)
        fL = f()
        RV = (fR-fL)/(2*h)
        self.setAB(a,self.b)
        return RV

    def numGrad_b(self,f,h=1e-6):
        b = self.b
        self.setAB(self.a,b+h)
        fR = f()
        self.setAB(self.a,b-h)
        fL = f()
        RV = (fR-fL)/(2*h)
        self.setAB(self.a,b)
        return RV

