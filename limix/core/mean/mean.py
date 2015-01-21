import sys
sys.path.insert(0,'./../../..')
from limix.core.cobj import * 
from limix.utils.preprocess import regressOut
import scipy as SP
import scipy.linalg as LA
import copy
import pdb

class mean(cObject):

    def __init__(self,Y):
        """ init data term """
        self.Y = Y
        self.clearFixedEffect()

    #########################################
    # Properties 
    #########################################

    @property
    def A(self):
        return self._A

    @property
    def B(self):
        return self._B

    @property
    def F(self):
        return self._F

    @property
    def Y(self):
        return self._Y

    @property
    def N(self):
        return self._N

    @property
    def P(self):
        return self._P

    @property
    def n_fixed_effs(self):
        return self._n_fixed_effs

    @property
    def n_terms(self):
        return self._n_terms

    @property
    def Lr(self):
        return self._Lr

    @property
    def Lc(self):
        return self._Lc

    @property
    def d(self):
        return self._d

    @property
    def D(self):
        return SP.reshape(self.d,(self.N,self.P), order='F')

    @property
    def LRLdiag(self):
        return self._LRLdiag

    @property
    def LCL(self):
        return self._LCL

    #########################################
    # Setters 
    #########################################

    def clearFixedEffect(self):
        """ erase all fixed effects """
        self._A = []
        self._F = []
        self._B = [] 
        self._n_terms = 0
        self._n_fixed_effs = 0
        self.indicator = {'term':SP.array([]),
                            'row':SP.array([]),
                            'col':SP.array([])}
        self.clear_cache('Fstar','Astar','Xstar','Xhat',
                         'Areml','Areml_chol','Areml_inv','beta_hat','B_hat',
                         'LRLdiag_Xhat_tens','Areml_grad',
                         'beta_grad','Xstar_beta_grad')

    def addFixedEffect(self,F=None,A=None):
        """
        set sample and trait designs
        F:  NxK sample design
        A:  LxP sample design
        """
        if F is None:   F = SP.ones((N,1))
        if A is None:   A = SP.eye(P)
        assert F.shape[0]==self.N, "F dimension mismatch"
        assert A.shape[1]==self.P, "A dimension mismatch"
        self.F.append(F)
        self.A.append(A)
        # build B matrix and indicator
        self.B.append(SP.zeros((F.shape[1],A.shape[0])))
        self._update_indicator(F.shape[1],A.shape[0])
        self._n_terms+=1
        self._n_fixed_effs+=F.shape[1]*A.shape[0]
        self.clear_cache('Fstar','Astar','Xstar','Xhat',
                         'Areml','Areml_chol','Areml_inv','beta_hat','B_hat',
                         'LRLdiag_Xhat_tens','Areml_grad',
                         'beta_grad','Xstar_beta_grad')

    @Y.setter
    def Y(self,value):
        """ set phenotype """
        self._N,self._P = value.shape
        self._Y = value
        self.clear_cache('Ystar1','Ystar','Yhat','LRLdiag_Yhat',
                         'beta_grad','Xstar_beta_grad')

    @Lr.setter
    def Lr(self,value):
        """ set row rotation """
        assert value.shape[0]==self._N, 'dimension mismatch'
        assert value.shape[1]==self._N, 'dimension mismatch'
        self._Lr = value
        self.clear_cache('Fstar','Ystar1','Ystar','Yhat','Xstar','Xhat',
                         'Areml','Areml_chol','Areml_inv','beta_hat','B_hat',
                         'LRLdiag_Xhat_tens','LRLdiag_Yhat','Areml_grad',
                         'beta_grad','Xstar_beta_grad',
                         'LRLdiag_Xhat_tens','LRLdiag_Yhat','Areml_grad',
                         'beta_grad','Xstar_beta_grad')

    @Lc.setter
    def Lc(self,value):
        """ set col rotation """
        assert value.shape[0]==self._P, 'Lc dimension mismatch'
        assert value.shape[1]==self._P, 'Lc dimension mismatch'
        self._Lc = value
        self.clear_cache('Astar','Ystar','Yhat','Xstar','Xhat',
                         'Areml','Areml_chol','Areml_inv','beta_hat','B_hat',
                         'LRLdiag_Xhat_tens','LRLdiag_Yhat','Areml_grad',
                         'beta_grad','Xstar_beta_grad')

    @d.setter
    def d(self,value):
        """ set anisotropic scaling """
        assert value.shape[0]==self._P*self._N, 'd dimension mismatch'
        self._d = value
        self.clear_cache('Yhat','Xhat','Areml','Areml_chol','Areml_inv','beta_hat','B_hat',
                         'LRLdiag_Xhat_tens','LRLdiag_Yhat','Areml_grad',
                         'beta_grad','Xstar_beta_grad')

    @LRLdiag.setter
    def LRLdiag(self,value):
        """ set anisotropic scaling """
        self._LRLdiag = value
        self.clear_cache('LRLdiag_Xhat_tens','LRLdiag_Yhat','Areml_grad',
                         'beta_grad','Xstar_beta_grad')

    @LCL.setter
    def LCL(self,value):
        """ set anisotropic scaling """
        self._LCL = value
        self.clear_cache('Areml_grad','beta_grad','Xstar_beta_grad')

    #########################################
    # Getters (caching)
    #########################################

    @cached
    def Astar(self):
        RV = []
        for term_i in range(self.n_terms):
            RV.append(SP.dot(self.A[term_i],self.Lc.T))
        return RV 

    @cached
    def Fstar(self):
        RV = []
        for term_i in range(self.n_terms):
            RV.append(SP.dot(self.Lr,self.F[term_i]))
        return RV 

    @cached
    def Ystar1(self):
        return SP.dot(self.Lr,self.Y)

    @cached
    def Ystar(self):
        return SP.dot(self.Ystar1(),self.Lc.T)

    @cached
    def Yhat(self):
        return self.D*self.Ystar()

    @cached
    def Xstar(self):
        RV = SP.zeros((self.N*self.P,self.n_fixed_effs))
        ip = 0
        for i in range(self.n_terms):
            Ki = self.A[i].shape[0]*self.F[i].shape[1]
            RV[:,ip:ip+Ki] = SP.kron(self.Astar()[i].T,self.Fstar()[i])
            ip += Ki
        return RV 

    @cached
    def Xhat(self):
        RV = self.d[:,SP.newaxis]*self.Xstar()
        return RV 

    @cached
    def Areml(self):
        return self.XstarT_dot(self.Xhat())

    @cached
    def Areml_chol(self):
        return LA.cholesky(self.Areml()).T

    @cached
    def Areml_inv(self):
        return LA.cho_solve((self.Areml_chol(),True),SP.eye(self.n_fixed_effs))

    @cached
    def beta_hat(self):
        return SP.dot(self.Areml_inv(),self.XstarT_dot(SP.reshape(self.Yhat(),(self.Y.size,1),order='F')))

    @cached
    def B_hat(self):
        RV = []
        ip = 0
        for term_i in range(self.n_terms):
            RV.append(SP.reshape(self.beta_hat()[ip:ip+self.B[term_i].size],self.B[term_i].shape, order='F'))
            ip += self.B[term_i].size
        return RV

    @cached
    def LRLdiag_Xhat_tens(self):
        RV  = SP.reshape(self.Xhat(),(self.N,self.P,self.n_fixed_effs),order='F').copy()
        RV *= self.LRLdiag[:,SP.newaxis,SP.newaxis]
        return RV

    @cached
    def LRLdiag_Yhat(self):
        return self.LRLdiag[:,SP.newaxis]*self.Yhat()

    @cached
    def Areml_grad(self):
        RV = SP.einsum('jpk,lp->jlk',self.LRLdiag_Xhat_tens(),self.LCL)
        RV = RV.reshape((self.N*self.P,self.n_fixed_effs),order='F')
        RV*= self.d[:,SP.newaxis]
        RV = -self.XstarT_dot(RV)
        return RV

    @cached
    def beta_grad(self):
        RV  = SP.reshape(SP.dot(self.LRLdiag_Yhat(),self.LCL.T),(self.N*self.P,1),order='F')
        RV *= self.d[:,SP.newaxis]
        RV  = self.XstarT_dot(RV)
        RV += SP.dot(self.Areml_grad(),self.beta_hat())
        RV  = -SP.dot(self.Areml_inv(),RV)
        return RV

    @cached
    def Xstar_beta_grad(self):
        RV = SP.zeros((self.N,self.P))
        ip = 0
        for term_i in range(self.n_terms):
            _Bgrad = SP.reshape(self.beta_grad()[ip:ip+self.B[term_i].size],self.B[term_i].shape, order='F')
            RV+=SP.dot(self.Fstar()[term_i],SP.dot(_Bgrad,self.Astar()[term_i]))
            ip += self.B[term_i].size
        return RV

    ###############################################
    # Other getters with no caching, should not they have caching somehow?
    ###############################################

    def predict(self):
        """ predict the value of the fixed effect """
        RV = SP.zeros((self.N,self.P))
        for term_i in range(self.n_terms):
            RV+=SP.dot(self.Fstar()[term_i],SP.dot(self.B()[term_i],self.Astar()[term_i]))
        return RV

    def evaluate(self):
        """ predict the value of """
        RV  = -self.predict()
        RV += self.Ystar()
        return RV

    def getGradient(self,j):
        """ get rotated gradient for fixed effect i """
        i = int(self.indicator['term'][j])
        r = int(self.indicator['row'][j])
        c = int(self.indicator['col'][j])
        rv = -SP.kron(self.Fstar()[i][:,[r]],self.Astar()[i][[c],:])
        return rv

    def XstarT_dot(self,M):
        """ get dot product of Xhat and M """
        if 0:
            #TODO: implement this properly
            pass
        else:
            RV = SP.dot(self.Xstar().T,M)
        return RV

    def Zstar(self):
        """ predict the value of the fixed effect """
        RV = self.Ystar().copy()
        for term_i in range(self.n_terms):
            RV-=SP.dot(self.Fstar()[term_i],SP.dot(self.B_hat()[term_i],self.Astar()[term_i]))
        return RV

    def getResiduals(self):
        """ regress out fixed effects and results residuals """
        X = SP.zeros((self.N*self.P,self.n_fixed_effs))
        ip = 0
        for i in range(self.n_terms):
            Ki = self.A[i].shape[0]*self.F[i].shape[1]
            X[:,ip:ip+Ki] = SP.kron(self.A[i].T,self.F[i])
            ip += Ki
        y = SP.reshape(self.Y,(self.Y.size,1),order='F')
        RV = regressOut(y,X)
        RV = SP.reshape(RV,self.Y.shape,order='F')
        return RV

    #########################################
    # Params manipulation 
    #########################################

    def getParams(self):
        """ get params """
        rv = SP.array([])
        if self.n_terms>0: 
            rv = SP.concatenate([SP.reshape(self.B[term_i],self.B[term_i].size, order='F') for term_i in range(self.n_terms)])
        return rv

    def setParams(self,params):
        """ set params """
        start = 0
        for i in range(self.n_terms):
            np = self.B[i].size
            self.B[i] = SP.reshape(params[start:start+np],self.B[i].shape, order='F')
            start += np

    #########################################
    # Utility functions
    #########################################

    def getDimensions(self):
        """ get phenotype dimensions """
        return self.N,self.P

    def _set_toChange(x):
        """ set variables in list x toChange """
        for key in x.keys():
            self.toChange[key] = True

    def _update_indicator(self,K,L):
        """ update the indicator """
        _update = {'term': self.n_terms*SP.ones((K,L)).T.ravel(),
                    'row': SP.kron(SP.arange(K)[:,SP.newaxis],SP.ones((1,L))).T.ravel(),
                    'col': SP.kron(SP.ones((K,1)),SP.arange(L)[SP.newaxis,:]).T.ravel()} 
        for key in _update.keys():
            self.indicator[key] = SP.concatenate([self.indicator[key],_update[key]])
