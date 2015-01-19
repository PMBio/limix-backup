import sys
sys.path.insert(0,'./..')
from cobj import * 
import scipy as SP
import copy
import pdb

class mean(cObject):

	def __init__(self,Y):
		""" init data term """
		self.Y = Y
		self._init()
		self.clearFixedEffect()

    #########################################
    # Properties 
    #########################################

    @property
    def A():
        return self._A

    @property
    def B():
        return self._B

    @property
    def F():
        return self._F

    @property
    def Y():
        return self._Y

    @property
    def Lr():
        return self._Lr

    @property
    def Lc():
        return self._Lc

    @property
    def d():
        return self._d

    @property
    def D():
        return SP.reshape(self.d,(self.N,self.P), order='F')

    #########################################
    # Setters 
    #########################################

	def clearFixedEffect(self):
		""" erase all fixed effects """
		self._A = []
		self._F = []
		self._B = [] 
		self.n_terms = 0
        self.n_fixed_effs = 0
		self.indicator = {'term':SP.array([]),
							'row':SP.array([]),
							'col':SP.array([])}
        self.clear_cache('Fstar','Astar','Xhat')

	def addFixedEffect(self,F=None,A=None):
		"""
		set sample and trait designs
		F:	NxK sample design
		A:	LxP sample design
		"""
		if F is None:	F = SP.ones((N,1))
		if A is None:	A = SP.eye(P)
		assert F.shape[0]==self.N, "F dimension mismatch"
		assert A.shape[1]==self.P, "A dimension mismatch"
		self.F.append(F)
		self.A.append(A)
		# build B matrix and indicator
		self.B.append(SP.zeros((F.shape[1],A.shape[0])))
		self._update_indicator(F.shape[1],A.shape[0])
		self.n_terms+=1
        self.n_fixed_effs+=F.shape[1]*A.shape[0]
        self.clear_cache('Fstar','Astar','Xhat')

    @Y.setter
	def Y(self,value):
		""" set phenotype """
		self._N,self.P = Y.shape
		self._Y = Y
        self.clear_cache('Ystar1','Ystar','Yhat')

    @Lr.setter
	def Lr(self,value):
		""" set row rotation """
	    assert value.shape[0]==self._N, 'dimension mismatch'
		assert value.shape[1]==self._N, 'dimension mismatch'
		self._Lr = value
        self.clear_cache('Fstar','Ystar1','Ystar','Yhat','Xhat')

    @Lc.setter
	def Lc(self,value):
		""" set col rotation """
        assert value.shape[0]==self._P, 'Lc dimension mismatch'
        assert value.shape[1]==self._P, 'Lc dimension mismatch'
		self._Lc = value
        self.clear_cache('Astar','Ystar','Yhat','Xhat')

    @d.setter
    def d(self,value):
        """ set anisotropic scaling """
        assert value.shape[0]==self._P*self._N, 'd dimension mismatch'
        self._d = value
        self.clear_cache('Yhat','Xhat')

    #########################################
    # Getters (caching)
    #########################################

    @cached
    def Astar(self):
        RV = []
        for term_i in range(self.n_terms):
            RV.append(SP.dot(self.getA(term_i),self.Lc.T))
        return RV 

    @cached
    def Fstar(self):
        RV = []
        for term_i in range(self.n_terms):
            RV.append(SP.dot(self.Lr,self.getF(term_i)))
        return RV 

    @cached
    def Ystar1(self):
        return SP.dot(self.Lr,self.Y)

    @cached
    def Ystar(self):
        return SP.dot(self.Ystar1(),self.Lc.T)

    @cached
    def getYhat(self):
        return self.D*self.getYstar()

    @cached
    def getXhat(self):
        RV = SP.zeros((N*P,self.n_fixed_effs))
        ip = 0
        for i in range(self.n_terms):
            Ki = self.A[i].shape[0]*self.F[i].shape[1]
            self.Xhat[:,ip:ip+Ki] = self.d[:,SP.newaxis]*SP.kron(self.Fstar[i],self.Astar[i].T)
            ip += Ki
        return RV 

    ###############################################
    # Other getters with no caching, should not they have caching somehow?
    ###############################################

	def predict(self):
		""" predict the value of the fixed effect """
		RV = SP.zeros((self.N,self.P))
		for term_i in range(self.n_terms):
			RV+=SP.dot(self.Fstar()[term_i],SP.dot(self.getB()[term_i],self.Astar()[term_i]))
		return RV

	def evaluate(self):
		""" predict the value of """
		RV  = -self.predict()
		RV += self.getYstar()
		return RV

	def getGradient(self,j):
		""" get rotated gradient for fixed effect i """
		i = int(self.indicator['term'][j])
		r = int(self.indicator['row'][j])
		c = int(self.indicator['col'][j])
		rv = -SP.kron(self.getFstar()[i][:,[r]],self.getAstar()[i][[c],:])
		return rv

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

    def _init(self):
        """ initializes all quantities """
        # rotations and scalings
        self.Lr = SP.eye(self.N)
        self.Lc = SP.eye(self.P)
        self.d  = SP.ones(self.N*self.P)
        # pheno transformations
        Ystar1 = Y.copy()
        Ystar  = Y.copy()
        Yhat   = Y.copy()
        # fixed effect transformations
        self.Astar = copy.copy(self.A)
        self.Fstar = copy.copy(self.F)
        # set toChange dict
        x = ['Y','Ystar','Ystar1','Fstar','Astar','Xhat']
        for key in x.keys():
            self.toChange[key] = False
        self.toChange['Xhat'] = True

	def _update_indicator(self,K,L):
		""" update the indicator """
		_update = {'term': self.n_terms*SP.ones((K,L)).T.ravel(),
					'row': SP.kron(SP.arange(K)[:,SP.newaxis],SP.ones((1,L))).T.ravel(),
					'col': SP.kron(SP.ones((K,1)),SP.arange(L)[SP.newaxis,:]).T.ravel()} 
		for key in _update.keys():
			self.indicator[key] = SP.concatenate([self.indicator[key],_update[key]])

