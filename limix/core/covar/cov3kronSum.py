import sys
sys.path.insert(0,'./../../..')
from limix.core.cobj import *
from limix.core.utils.eigen import *
import scipy as SP
import scipy.linalg as LA
import scipy.sparse.linalg as SLA
import scipy.stats.mstats as MST
import warnings
from covar_base import Covariance

from scipy.optimize import fmin_l_bfgs_b as optimize

import pdb

# should be a child class of combinator
class cov3kronSum(Covariance):

    def __init__(self,C1=None,C2=None,Cn=None,R1=None,R2=None,gradient_method='lb',nIterMC=30,tol=1E-3,linsys_method='rot'):
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

        self.tol = tol
        self.gradient_method = 'lb'
        self.linsys_method   = 'std'
        self.nIterMC = nIterMC
        self.drawZ()

    def setGradientMethod(self,gradient_method):
        self.gradient_method = gradient_method

    def setLinSysMethod(self,linsys_method):
        self.linsys_method = linsys_method

    def setTolerance(self,tolerance):
        self.tolerance = tolerance

    def setParams(self,params):
        self.C1.setParams(params[:self.C1.getNumberParams()])
        self.C2.setParams(params[self.C1.getNumberParams():self.C1.getNumberParams()+self.C2.getNumberParams()])
        self.Cn.setParams(params[self.C1.getNumberParams()+self.C2.getNumberParams():])
        self.clear_cache('C3','S_C3','U_C3','USi2_C3')
        self.clear_cache('C1star','S_C1star','U_C1star')
        self.clear_cache('C2star','S_C2star','U_C2star')

        self.clear_cache('Cn','S_Cn','U_Cn','USi2_Cn')
        self.clear_cache('C1star_n','S_C1star_n','U_C1star_n')
        self.clear_cache('C2star_n','S_C2star_n','U_C2star_n')

        self.clear_cache('logdet','logdet_bound')
        self.clear_cache('K','Kinv','solveKinvZ')

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
        self.clear_cache('K','Kinv','solveKinvZ')
        self.clear_cache('R1Z','UR2t_R1_UR2')

    def setR2(self,R2):
        self.R2 = R2
        self.clear_cache('S_R2','U_R2')
        self.clear_cache('logdet','logdet_bound')
        self.clear_cache('K','Kinv','solveKinvZ')
        self.clear_cache('R2Z','UR2t_R1_UR2')

    def getParams(self):
        return SP.concatenate([self.C1.getParams(),self.C2.getParams(),self.Cn.getParams()])

    def _calcNumberParams(self):
        self.n_params = self.C1.getNumberParams()+self.C2.getNumberParams()+self.Cn.getNumberParams()


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

    def _KxRot(self,x):
        """
        compute Krot x
        """

        #TODO: put Cstar in a separate function
        Cstar = SP.dot(self.U_C2star_n().T,SP.dot(self.C1star_n(),self.U_C2star_n()))
        X = SP.reshape(x,(self.N,self.P),order='F')
        Y = SP.dot(self.UR2t_R1_UR2(),SP.dot(X,Cstar.T))
        Y+= self.S_C2star_n()*(X.T*self.S_R2()).T
        Y+= X
        y = SP.reshape(Y,(self.N*self.P),order='F')
        return y

    def solve(self,Y,X0=None,tol=None):
        if self.linsys_method=='std':
            # no rotation is used
            return self.solve_std(Y,X0=X0,tol=tol)
        elif self.linsys_method=='rot':
            # rotating out one component
            return self.solve_rot(Y,X0=X0,tol=tol)


    def solve_rot(self,Y,X0=None,tol=None):
        """ solve lin system Ki Y = X by whitening out first """
        if tol is None: tol = self.tol

        Kx_linop = SLA.LinearOperator((self.N*self.P,self.N*self.P),matvec=self._KxRot,rmatvec=self._Kx,dtype='float64')
        x0 = SP.reshape(X0,(self.N*self.P),order='F')

        # rotate input
        # TODO: put Cstar in a separate function
        Cstar = SP.dot(self.USi2_Cn(),self.U_C2star_n())

        # TODO: ideally, we would like to precompute U_R2*Y for certain Ys
        Yrot = SP.dot(self.U_R2().T,SP.dot(Y,Cstar))

        # solve linear system
        yrot = SP.reshape(Yrot,(self.N*self.P),order='F')
        ytilde,_ = SLA.cgs(Kx_linop,yrot,x0=x0,tol=self.tol)
        Ytilde = SP.reshape(ytilde,(self.N,self.P),order='F')

        # rotate output
        RV = SP.dot(self.U_R2(),SP.dot(Ytilde,Cstar.T))
        return RV,Ytilde

    def solve_std(self,Y,tol=None,X0=None):
        """ solve lin system Ki Y = X """
        if tol is None: tol = self.tol
        Kx_linop = SLA.LinearOperator((self.N*self.P,self.N*self.P),matvec=self._Kx,rmatvec=self._Kx,dtype='float64')
        x0 = SP.reshape(X0,(self.N*self.P),order='F')
        y  = SP.reshape(Y,(self.N*self.P),order='F')
        x,_ = SLA.cgs(Kx_linop,y,x0=x0,tol=self.tol)
        RV = SP.reshape(x,(self.N,self.P),order='F')
        return RV,RV

    def logdet_grad_1(self,i):
        if self.gradient_method=='exact':
            return self.logdet_exact_grad_1(i)
        elif self.gradient_method=='lb':
            return self.logdet_bound_grad_1(i)
        elif self.gradient_method=='mc':
            return self.logdet_mc_grad_1(i)

    def logdet_grad_2(self,i):
        if self.gradient_method=='exact':
            return self.logdet_exact_grad_2(i)
        elif self.gradient_method=='lb':
            return self.logdet_bound_grad_2(i)
        elif self.gradient_method=='mc':
            return self.logdet_mc_grad_2(i)

    def logdet_grad_n(self,i):
        if self.gradient_method=='exact':
            return self.logdet_exact_grad_n(i)
        elif self.gradient_method=='lb':
            return self.logdet_bound_grad_n(i)
        elif self.gradient_method=='mc':
            return self.logdet_mc_grad_n(i)

    ###########################
    # exact computations
    ##########################

    @cached
    def K(self):
        assert self.R2.shape[0]*self.C1.K().shape[0]<5000, 'dimension is too high'
        RV = SP.kron(self.C1.K(),self.R1)
        RV+= SP.kron(self.C2.K(),self.R2)
        RV+= SP.kron(self.Cn.K(),SP.eye(self.N))
        return RV

    @cached
    def Kinv(self):
        return LA.inv(self.K())

    @cached
    def logdet(self):
        S,U = LA.eigh(self.K())
        return SP.log(S).sum()

    def logdet_exact_grad_1(self,i):
        assert self.R2.shape[0]*self.C1.K().shape[0]<5000, 'dimension is too high'
        Kgrad_i = SP.kron(self.C1.Kgrad_param(i),self.R1)
        return SP.sum(Kgrad_i*self.Kinv())


    def logdet_exact_grad_2(self,i):
        assert self.R2.shape[0]*self.C1.K().shape[0]<5000, 'dimension is too high'
        Kgrad_i = SP.kron(self.C2.Kgrad_param(i),self.R2)
        return SP.sum(Kgrad_i*self.Kinv())


    def logdet_exact_grad_n(self,i):
        assert self.R2.shape[0]*self.C1.K().shape[0]<5000, 'dimension is too high'
        Kgrad_i = SP.kron(self.Cn.Kgrad_param(i),SP.eye(self.N))
        return SP.sum(Kgrad_i*self.Kinv())



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

    @cached
    def UR2t_R1_UR2(self):
        return SP.dot(self.U_R2().T,SP.dot(self.R1,self.U_R2()))

    ###########################
    # Cn
    ##########################

    @cached
    def S_Cn(self):
        S,U = LA.eigh(self.Cn.K())
        self.fill_cache('U_Cn',U)
        return S

    @cached
    def U_Cn(self):
        S,U = LA.eigh(self.Cn.K())
        self.fill_cache('S_Cn',S)
        return U

    @cached
    def USi2_Cn(self):
        return self.U_Cn()*self.S_Cn()**(-0.5)

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
    def C1star_n(self):
        return SP.dot(self.USi2_Cn().T,SP.dot(self.C1.K(),self.USi2_Cn()))

    @cached
    def S_C1star_n(self):
        S,U = LA.eigh(self.C1star_n())
        self.fill_cache('U_C1star_n',U)
        return S

    @cached
    def U_C1star_n(self):
        S,U = LA.eigh(self.C1star_n())
        self.fill_cache('S_C1star_n',S)
        return U


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
    def C2star_n(self):
        return SP.dot(self.USi2_Cn().T,SP.dot(self.C2.K(),self.USi2_Cn()))

    @cached
    def S_C2star_n(self):
        S,U = LA.eigh(self.C2star_n())
        self.fill_cache('U_C2star_n',U)
        return S

    @cached
    def U_C2star_n(self):
        S,U = LA.eigh(self.C2star_n())
        self.fill_cache('S_C2star_n',S)
        return U


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
        if 0:
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




    ###########################
    # Monte Carlo Trace Approximation
    ##########################

    def set_nIterMC(self,nIterMC):
        self.nIterMC = nIterMC

    def drawZ(self,seed=0):
        rs = SP.random.RandomState(seed)
        self.Z  = rs.randn(self.nIterMC,self.N,self.P)
        self.KinvZ  = SP.zeros((self.nIterMC,self.N,self.P))
        self.KinvZ0 = SP.zeros((self.nIterMC,self.N,self.P))
        self.clear_cache('R1Z','R2Z','solveKinvZ')

    @cached
    def R1Z(self):
        return SP.transpose(SP.tensordot(self.R1,self.Z,axes=(1,1)),axes=(1,0,2))

    @cached
    def R2Z(self):
        return SP.transpose(SP.tensordot(self.R2,self.Z,axes=(1,1)),axes=(1,0,2))

    @cached
    def solveKinvZ(self):
        for j in range(self.nIterMC):
            self.KinvZ[j],self.KinvZ0[j] = self.solve(self.Z[j],X0=self.KinvZ0[j])
        return self.KinvZ

    def logdet_mc_grad_1(self,i):
        KgradZ = SP.dot(self.R1Z(),self.C1.Kgrad_param(i))
        KinvZ = self.solveKinvZ()
        return SP.sum(KgradZ*KinvZ)/self.nIterMC

    def logdet_mc_grad_2(self,i):
        KinvZ = self.solveKinvZ()
        KgradZ = SP.dot(self.R2Z(),self.C2.Kgrad_param(i))
        return SP.sum(KgradZ*KinvZ)/self.nIterMC

    def logdet_mc_grad_n(self,i):
        KgradZ =  SP.dot(self.Z,self.Cn.Kgrad_param(i))
        KinvZ = self.solveKinvZ()
        return SP.sum(KgradZ*KinvZ)/self.nIterMC
