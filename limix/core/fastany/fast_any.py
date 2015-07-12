import sys
import ipdb
import numpy.linalg as la 
import numpy as np
import scipy as sp
import limix.core.mean.mean as mean
from limix.core.association.kron_util import *

class GeneratorKron(object):
	def __init__(self,N=100,S=5,R=1000,P=3,var_K1=0.6,var_K2=0.4,h2_1=0.9,h2_2=0.1,A=None, var_snps=None,h2=0.5):
		assert var_K1>=0, "var_K1 has to be greater or equal to zero"
		assert var_K2>=0, "var_K2 has to be greater or equal to zero"
		assert h2_1<=1, "h2_1 has to be smaller or equal to one"
		assert h2_1>=0, "h2_1 has to be larger or equal to 0"
		assert h2_2<=1, "h2_1 has to be smaller or equal to one"
		assert h2_2>=0, "h2_1 has to be larger or equal to 0"

		self.A = A
		self.N = N
		self.P = P
		self.R = R

		self.debug = True

		self.h2_1 = h2_1
		self.h2_2 = h2_2
		self.var_K1 = var_K1
		self.var_K2 = var_K2

		if var_snps is None:
			self.h2 = h2
		else:
			assert var_snps>=0.0
			self.var_snps = var_snps

		self.generateR1()
		self.generateR2()
		self.generateC1()
		self.generateC2()
		self.generate_snps(S)
		self.generate_phenotype()
	
	@property
	def h2(self):
		return self.var_snps/(self.var_total)

	@h2.setter
	def h2(self,h2):
		assert h2>=0.0
		assert h2<=1.0
		self.var_snps = h2/(1-h2) * (self.var_K1 + self.var_K2)		

	@property
	def var(self):
		return self.var_K1+self.var_K2
	
	@property
	def var_total_empirical(self):
		return (np.sqrt(self.var_snps) * self.Y_snps + self.Y_back).var()

	@property
	def var_empirical(self):
		return self.Y_back.var()

	@property
	def var_total(self):
		return self.var+self.var_snps

	@property
	def Y_back(self):
		return np.sqrt(self.var_K1) * self.Y1 + np.sqrt(self.var_K2) * self.Y2
	
	@property
	def Y(self):
		return np.sqrt(self.var_snps) * self.Y_snps + self.Y_back + self.mean

	def generate_phenotype(self):
		self.generate_snp_effects()
		self.generate_background()
		self.generate_mean()
		return self.Y

	def generate_mean(self):
		mean = np.random.randn(self.P)+np.random.normal()
		self.mean = mean[np.newaxis,:]
		return self.mean

	def generate_background(self):
		self.Y1 = generate_matrix_normal(self.R1,self.C1)
		self.Y2 = generate_matrix_normal(self.R2,self.C2)
		return self.Y1,self.Y2
	
	def generate_snp_effects(self):
		self.Y_snps = generate_snp_effects_kron(snps=self.snps,P=self.P,N=self.N,A=self.A,standardize=True)
		return self.Y_snps

	def Rrot(self):
		return rot_kron(self.R1,self.R2)

	def Crot(self):
		return rot_kron(self.C1,self.C2)

	def Yrot(self):
		res = self.Rrot().dot(self.Y).dot(self.Crot())
		return res

	def snps_rot(self):
		res = self.Rrot().dot(self.snps)
		return res
			

	@property
	def S(self):
		return self.snps.shape[1]
	 
	def generate_snps(self,S):
		self.snps = np.random.randn(self.N,S)
		
	def generateR1(self):
		self.R1,self.X1 = generate_kernel(N=self.N, R=self.R, h2=self.h2_1)

	def generateR2(self):
		self.R2,self.X2 = generate_kernel(N=self.N, R=self.R, h2=self.h2_2)

	def generateC1(self):
		self.C1,self.A1 = generate_kernel(N=self.P, R=self.R, h2=self.h2_1)

	def generateC2(self):
		self.C2,self.A2 = generate_kernel(N=self.P, R=self.R, h2=self.h2_2)

def generate_kernel(N,R=100, h2=0.99999999):
	assert h2>=0.0
	assert h2<=1.0
	X = np.random.randn(N,R)#random effects
	XX = X.dot(X.T)
	XX /= XX.diagonal().mean()
	R = h2 * XX + (1-h2) * np.eye(N)
	return R, X

def generate_snp_effects_kron(snps,P=1,N=100,A=None, standardize=True, eps=1E-8):
	assert eps<=1.0
	assert eps>=0.0
	if A is None:
		A = np.eye(P)
	XX = (1.0-eps)*snps.dot(snps.T)+eps*np.eye(N)
	if standardize:
		XX/=XX.diagonal().mean()
	AA = (1.0-eps)*A.dot(A.T)+eps*np.eye(P)
	if standardize:
		AA/=AA.diagonal().mean()
	Y = generate_matrix_normal(R=XX,C=AA)
	return Y

def generate_snp_effects_primal(snps,P=1,A=None):
	if A is None:
		A = np.eye(P)
	beta = np.random.randn(snps.shape[1],A.shape[1])
	Y = snps.dot(beta.dot(A.T))
	return beta,Y

 

