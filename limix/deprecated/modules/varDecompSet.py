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
import limix.deprecated.modules.qtl as qtl
import limix.deprecated.stats.fdr as fdr
from limix.deprecated.stats.pca import *
import limix.deprecated as dlimix
import scipy as sp
import scipy.linalg as la
import pdb
import scipy.linalg as linalg
import time

class varDecompSet:
    """variance decomposition set class"""
    def __init__(self,Y=None,Ks=None,standardize=True,noise=None):
        """
        Args:
            X: alternatively SNP data
            Y: alternativel expression data
            Ks: list of covariance matrices
            standardize: if True, phenotypes are standardized
        """
        assert Y is not None, 'Specify Y'
        assert Ks is not None, 'Specify Ks'
        if type(Ks)!=list: Ks = [Ks]
        self.Y  = Y.copy()
        self.Ks = Ks
        if standardize:
            self.Y -= self.Y.mean(axis=0)
            self.Y /= self.Y.std(axis=0)
        self.N = self.Y.shape[0]
        self.P = self.Y.shape[1]
        self.noise = noise
        self.n_terms = len(self.Ks)+1
        if self.noise is 'off':
            self.n_terms-=1

    def train(self,jitter=1e-4):
        """train vds module"""
        covar  = limix.CSumCF()
        covar_params = []
        for K in self.Ks:
            covar.addCovariance(limix.CFixedCF(K+jitter*sp.eye(self.N)))
            covar_params.append(sp.ones(1)/sp.sqrt(self.n_terms))
        if self.noise is 'correlated':
            _cov1 = limix.CFixedCF(sp.eye(int(self.Y.shape[0]/2)))
            self.cov_noise = limix.CFixedDiagonalCF(limix.CFreeFormCF(2),sp.ones(2))
            self.cov_noise.setParamMask(sp.array([0,1,0]))
            self.Knoise = limix.CKroneckerCF(_cov1,self.cov_noise)
            covar.addCovariance(self.Knoise)
            covar_params.append(sp.ones(1)/sp.sqrt(self.n_terms))
            covar_params.append(sp.array([1.,1e-3,1.]))
        elif self.noise is not 'off':
            covar.addCovariance(limix.CFixedCF(sp.eye(self.N)))
            covar_params.append(sp.ones(1)/sp.sqrt(self.n_terms))

        hyperparams = limix.CGPHyperParams()
        covar_params = sp.concatenate(covar_params)
        hyperparams['covar'] = covar_params
        constrainU = limix.CGPHyperParams()
        constrainL = limix.CGPHyperParams()
        constrainU['covar'] = +5*sp.ones_like(covar_params);
        constrainL['covar'] = -5*sp.ones_like(covar_params);

        self.gp=limix.CGPbase(covar,limix.CLikNormalNULL())
        self.gp.setY(self.Y)
        lml0 = self.gp.LML(hyperparams)
        dlml0 = self.gp.LMLgrad(hyperparams)
        gpopt = limix.CGPopt(self.gp)
        gpopt.setOptBoundLower(constrainL);
        gpopt.setOptBoundUpper(constrainU);

        t1 = time.time()
        conv = gpopt.opt()
        t2 = time.time()

        RV = {'Converged': True, 'time': t2-t1}
        return RV

    def getVarianceComps(self):
        """
        Returns:
            vector of variance components of the PANAMA, Kpop and noise contributions
        """
        if self.noise is 'correlated':
            RV = self.gp.getParams()['covar'][:self.n_terms-1]**2
        else:
            RV = self.gp.getParams()['covar']**2
        return RV

    def getNoise(self):
        assert self.noise is 'correlated', 'work only if noise is \'correlated\'!'
        return self.gp.getParams()['covar'][self.n_terms-1]**2*self.cov_noise.K()
