import scipy as SP
import pdb

class mean():

	def __init__(self,Y):
		""" init data term """
		self.setY(Y)
		self.clearFixedEffect()
		self.setRowRotation()
		self.setColRotation()

	def clearFixedEffect(self):
		""" erase all fixed effects """
		self.A = []
		self.F = []
		self.B = [] 
		self.Ar = []
		self.Fr = []
		self.n_terms = 0
		self.indicator = {'term':SP.array([]),
							'row':SP.array([]),
							'col':SP.array([])}
		self.designs_have_changed=True

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
		self.F.append(F); self.Fr.append(SP.zeros_like(F))
		self.A.append(A); self.Ar.append(SP.zeros_like(A))
		# build B matrix and indicator
		self.B.append(SP.zeros((F.shape[1],A.shape[0])))
		self._update_indicator(F.shape[1],A.shape[0])
		self.n_terms+=1
		self.designs_have_changed=True

	def predict(self):
		""" predict the value of the fixed effect """
		self._rotate()
		RV = SP.zeros((self.N,self.P))
		for term_i in range(self.n_terms):
			RV+=SP.dot(self.Fr[term_i],SP.dot(self.B[term_i],self.Ar[term_i]))
		return RV

	def evaluate(self):
		""" predict the value of """
		RV  = -self.predict()
		RV += self.Yr
		return RV

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

	def getDimensions(self):
		""" get phenotype dimensions """
		return self.N,self.P

	def setY(self,Y):
		""" set phenotype """
		self.N,self.P = Y.shape
		self.Y = Y
		self.Y_has_changed=True

	def setRowRotation(self,Lr=None):
		""" set row rotation """
		if Lr is not None:
			assert Lr.shape[0]==self.N, 'Lr dimension mismatch'
			assert Lr.shape[1]==self.N, 'Lr dimension mismatch'
		self.Lr = Lr
		self.Lr_has_changed = True

	def setColRotation(self,Lc=None):
		""" set col rotation """
		if Lc is not None:
			assert Lc.shape[0]==self.P, 'Lc dimension mismatch'
			assert Lc.shape[1]==self.P, 'Lc dimension mismatch'
		self.Lc = Lc
		self.Lc_has_changed = True

	def getGradient(self,j):
		""" get rotated gradient for fixed effect i """
		self._rotate()
		i = int(self.indicator['term'][j])
		r = int(self.indicator['row'][j])
		c = int(self.indicator['col'][j])
		rv = -SP.kron(self.Fr[i][:,[r]],self.Ar[i][[c],:])
		return rv

	def _rotate(self):
		""" rotate before rows using Lr and then rows using Lc """
		if self.Lr_has_changed or self.Y_has_changed:
			if self.Lr is not None:	self.Yr1 = SP.dot(self.Lr,self.Y)
			else:					self.Yr1 = self.Y.copy()

		if self.Lr_has_changed or self.designs_have_changed:
			for term_i in range(self.n_terms):
				if self.Lr is not None:	self.Fr[term_i] = SP.dot(self.Lr,self.F[term_i])
				else:					self.Fr[term_i] = self.F[term_i].copy()

		if self.Lc_has_changed or self.Y_has_changed:
			if self.Lc is not None:	self.Yr = SP.dot(self.Yr1,self.Lc.T)
			else:					self.Yr = self.Yr1.copy()

		if self.Lc_has_changed or self.designs_have_changed:
			for term_i in range(self.n_terms):
				if self.Lc is not None:	self.Ar[term_i] = SP.dot(self.A[term_i],self.Lc.T)
				else:					self.Ar[term_i] = self.A[term_i].copy()

		"""
		in principle here you can also set D and cache staff for DY 
		not sure this is useful
		"""

		self.Lr_has_changed = False
		self.Lc_has_changed = False
		self.Y_has_changed  = False
		self.designs_have_changed = False

	def _update_indicator(self,K,L):
		""" update the indicator """
		_update = {'term': self.n_terms*SP.ones((K,L)).T.ravel(),
					'row': SP.kron(SP.arange(K)[:,SP.newaxis],SP.ones((1,L))).T.ravel(),
					'col': SP.kron(SP.ones((K,1)),SP.arange(L)[SP.newaxis,:]).T.ravel()} 
		for key in _update.keys():
			self.indicator[key] = SP.concatenate([self.indicator[key],_update[key]])

