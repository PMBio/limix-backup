import sys
sys.path.append('./../build/src/python_interface')
sys.path.append('./../pygp')


import gpmix
import pygp.covar.linear as lin
import scipy as SP
import pdb

X = SP.randn(10,3)

params = SP.array([0])

c1 = gpmix.CCovLinearISO()
c2 = lin.LinearCFISO(n_dimensions=3)


c1.setX(X)

K1 = c1.K()
K2 = c2.K(params,X,X)

dK1 = c1.Kgrad_param(0)
dK2 = c2.Kgrad_theta(params,X,0)

dKx1= c1.Kgrad_X(0)
dKx2= c2.Kgrad_x(params,X,X,0)


dKx1diag = c1.Kdiag_grad_X(0)
dKx2diag = c2.Kgrad_xdiag(params,X,0)
