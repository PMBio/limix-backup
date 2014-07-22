# Copyright(c) 2014, The LIMIX developers (Christoph Lippert, Paolo Francesco Casale, Oliver Stegle)
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import sys
sys.path.append('./..')
sys.path.append('./../../build/src/interfaces/python')
import limix
import scipy as SP


# LINEAR COVARIANCE MATRIX
C = limix.CCovLinearARD(2)
"""
# NON-COMPATIBLE PARAM BOUNDS
# 1. Wrong number of components
LB=SP.array([0.])
UB=float('inf')*SP.ones(2)
C.setParamBounds(LB,UB)
# 2. Upper < Lower
LB=+1.*SP.ones(2)
UB=-1.*SP.ones(2)
C.setParamBounds(LB,UB)
"""
LB=SP.array([0.,5.])
UB=SP.array([float('inf'),10.])
C.setParamBounds(LB,UB)
print C.getParamBounds0()
print C.getParamBounds()


# MULTI COVARIANCE
C1 = limix.CFixedCF()
C2 = limix.CCovSqexpARD(1)
CP = limix.CProductCF()
CP.addCovariance(C1)
CP.addCovariance(C2)
CP.setParamBounds(5*SP.ones(3),10*SP.ones(3))
print CP.getParamBounds0()
print CP.getParamBounds()

# ALSO DEFINED IN LIKELIHOODS
L=limix.CLikNormalIso()
L.setParamBounds(SP.array([0]),SP.array([10]))
L.getParamBounds()

