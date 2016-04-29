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

"""
PANAMA module in limix
"""

import limix.deprecated
import limix.deprecated.stats.fdr as fdr
from limix.deprecated.stats.pca import *
from . import qtl
import scipy as sp 
import pdb
import scipy.linalg as linalg
import time

class PANAMA:
    """PANAMA class"""
    def __init__(self,data=None,X=None,Y=None,Kpop=None,use_Kpop=True,standardize=True):
        """
        Args:
            data: data object to feed form
            X: alternatively SNP data
            Y: alternativel expression data 
            Kpop: Kpop
            use_Kpop: if True (default), Kpop is considered in the model
            standardize: if True, phenotypes are standardized
        """
        self.use_Kpop = use_Kpop

        if data is not None:
            self.Kpop = data.getCovariance()
            self.Y = data.getPhenotypes()[0].copy()
            self.X = data.gtGenotypes()
        else:
            self.X = X
            self.Y = Y
        if Kpop is not None:
            self.Kpop = Kpop
        elif self.X is not None:
            self.Kpop = sp.dot(self.X,self.X.T)
            self.Kpop /= self.Kpop.diagonal().mean()
        else:
            assert use_Kpop==False, 'no Kpop'
        
        if standardize:
            self.Y -= self.Y.mean(axis=0)
            self.Y /= self.Y.std(axis=0)
        self.N = self.Y.shape[0]
        self.P = self.Y.shape[1]
        if self.X is not None:
            self.S = self.X.shape[1]
            assert self.X.shape[0]==self.Y.shape[0], 'data size missmatch'
        pass

    def train(self,rank=20,Kpop=True,LinearARD=False):
        """train panama module"""

        if 0:
            covar  = limix.deprecarted.CCovLinearISO(rank)
            ll  = limix.deprecated.CLikNormalIso()
            X0 = sp.random.randn(self.N,rank)
            X0 = PCA(self.Y,rank)[0]
            X0 /= sp.sqrt(rank)
            covar_params = sp.array([1.0])
            lik_params = sp.array([1.0])

            hyperparams = limix.deprecated.CGPHyperParams()
            hyperparams['covar'] = covar_params
            hyperparams['lik'] = lik_params
            hyperparams['X']   = X0
        
            constrainU = limix.deprecated.CGPHyperParams()
            constrainL = limix.deprecated.CGPHyperParams()
            constrainU['covar'] = +5*sp.ones_like(covar_params);
            constrainL['covar'] = 0*sp.ones_like(covar_params);
            constrainU['lik'] = +5*sp.ones_like(lik_params);
            constrainL['lik'] = 0*sp.ones_like(lik_params);

        if 1:
            covar  = limix.deprecated.CSumCF()
            if LinearARD:
                covar_1 =  limix.deprecated.CCovLinearARD(rank)
                covar_params = []
                for d in range(rank):
                    covar_params.append(1/sp.sqrt(d+2))
            else:
                covar_1 =  limix.deprecated.CCovLinearISO(rank)
                covar_params = [1.0]
            covar.addCovariance(covar_1)

            if self.use_Kpop:
                covar_2 =  limix.deprecated.CFixedCF(self.Kpop)
                covar.addCovariance(covar_2)
                covar_params.append(1.0)

            ll  = limix.deprecated.CLikNormalIso()
            X0 = PCA(self.Y,rank)[0]
            X0 /= sp.sqrt(rank)
            covar_params = sp.array(covar_params)
            lik_params = sp.array([1.0])

            hyperparams = limix.deprecated.CGPHyperParams()
            hyperparams['covar'] = covar_params
            hyperparams['lik'] = lik_params
            hyperparams['X']   = X0
        
            constrainU = limix.deprecated.CGPHyperParams()
            constrainL = limix.deprecated.CGPHyperParams()
            constrainU['covar'] = +5*sp.ones_like(covar_params);
            constrainL['covar'] = -5*sp.ones_like(covar_params);
            constrainU['lik'] = +5*sp.ones_like(lik_params);

            
        gp=limix.deprecated.CGPbase(covar,ll)
        gp.setY(self.Y)
        gp.setX(X0)
        lml0 = gp.LML(hyperparams)
        dlml0 = gp.LMLgrad(hyperparams)        
        gpopt = limix.deprecated.CGPopt(gp)
        gpopt.setOptBoundLower(constrainL);
        gpopt.setOptBoundUpper(constrainU);

        t1 = time.time()
        gpopt.opt()
        t2 = time.time()

        #Kpanama
        self.Xpanama = covar_1.getX()
        if LinearARD:
            self.Xpanama /= self.Xpanama.std(0)
        self.Kpanama = covar_1.K()
        self.Kpanama/= self.Kpanama.diagonal().mean()

        # Ktot
        self.Ktot = covar_1.K()
        if self.use_Kpop:
            self.Ktot += covar_2.K()
        self.Ktot/= self.Ktot.diagonal().mean()

        #store variances
        V = {}
        if LinearARD:
            V['LinearARD'] = covar_1.getParams()**2*covar_1.getX().var(0)
        else:
            V['Kpanama'] = sp.array([covar_1.K().diagonal().mean()])
        if self.use_Kpop:
            V['Kpop'] = sp.array([covar_2.K().diagonal().mean()])
        V['noise'] = gp.getParams()['lik']**2
        self.varianceComps = V

    def get_Xpanama(self):
        """
        Returns:
            matrix of Xs
        """
        return self.Xpanama
    
    def get_Kpanama(self):
        """
        Returns:
            Kpanama (normalized XX.T)
        """
        return self.Kpanama
    
    def get_Ktot(self):
        """
        Returns:
            Ktot (normalized Kpanama+Kpop)
        """
        return self.Ktot
    
    def get_varianceComps(self):
        """
        Returns:
            vector of variance components of the PANAMA, Kpop and noise contributions
        """
        return self.varianceComps

