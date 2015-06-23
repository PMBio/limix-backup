# Copyright(c) 2014, The LIMIX developers (Christoph Lippert, Paolo Francesco Casale, Oliver Stegle)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
sys.path.insert(0,'../..')
import scipy as sp 
import scipy.linalg
import scipy.stats
import limix
import limix.core.optimize.optimize_bfgs as OPT
import limix.core.mean
import limix.core.covar
import limix.core.gp
import pdb
import time
import copy
import warnings

try:
	from fastlmm.inference.lmm_cov import LMM as fastLMM
	fastlmm_present = True
except:
	fastlmm_present = False

class VarianceDecomposition:
    """
    Variance decomposition module in LIMIX
    This class mainly takes care of initialization and eases the interpretation of complex variance decompositions

    vc = varianceDecomposition(Y) # set phenotype matrix Y [N,P]
    vc.addFixedEffect() # set intercept
    vc.addRandomEffectTerm(K=K) # set genetic random effect with genetic kinship as sample covariance matrix [N,N]
    vc.addRandomEffectTerm(is_noise=True) # set noisy random effect
    vc.optimize() # train the gaussian process 
    vc.getTraitCovar(0) # get estimated trait covariance for random effect 1 [P,P]
    vc.getVarianceComps() # get variance components of the different terms for different traits as [P,n_randEffs]
    """
    
    def __init__(self,Y,standardize=False):
        """
        Args:
            Y:              phenotype matrix [N, P]
            standardize:    if True, impute missing phenotype values by mean value,
                            zero-mean and unit-variance phenotype (Boolean, default False)
        """

        #check whether Y is a vector, if yes reshape
        if (len(Y.shape)==1):
            Y = Y[:,sp.newaxis]
        
        #create column of 1 for fixed if nothing providede
        self.N       = Y.shape[0]
        self.P       = Y.shape[1]
        self.Nt      = self.N*self.P
        self.Iok     = ~(sp.isnan(Y).any(axis=1))

        #outsourced to handle missing values:
        if standardize:
            Y=preprocess.standardize(Y)

        # init term and covariances
        self.Y = Y
        self.mu = limix.core.mean.mean(self.Y)

        # random effs
        self.n_randEffs = 0
        self.R  = []
        self.C  = []
        self.Rstar = []
        self.Ntest = None

        self.gp      = None
        self.init    = False
        self.fast    = False
        self.noisPos = None

        self.cache = {}
        self.cache['Sigma']   = None
        self.cache['Hessian'] = None

        pass
    
    def setY(self,Y,standardize=False):
        """
        Set phenotype matrix
        
        Args:
            Y:              phenotype matrix [N, P]
            standardize:    if True, phenotype is standardized (zero mean, unit variance)
        """
        print "DEPRECATED"

        assert Y.shape[0]==self.N, 'VarianceDecomposition:: Incompatible shape'
        assert Y.shape[1]==self.P, 'VarianceDecomposition:: Incompatible shape'

        
        if standardize:
            Y=preprocess.standardize(Y)

        #check that missing values match the current structure
        assert (~(sp.isnan(Y).any(axis=1))==self.Iok).all(), 'VarianceDecomposition:: pattern of missing values needs to match Y given at initialization'

        self.Y = Y

        self.init = False
    
        self.cache['Sigma']   = None
        self.cache['Hessian'] = None

    def addRandomEffect(self,K=None,is_noise=False,normalize=True,Kcross=None,trait_covar_type='freeform',fixed_trait_covar=None,rank=None):
        """
        Add random effects term.
        
        Args:
            K:      Sample Covariance Matrix [N, N]
            is_noise:   Boolean indicator specifying if the matrix is homoscedastic noise (weighted identity covariance) (default False)
            normalize:  Boolean indicator specifying if K has to be normalized such that K.trace()=N.
            Kcross:            NxNtest cross covariance for predictions
            trait_covar_type: type of covaraince to use. Default 'freeform'. possible values are 
                            'freeform':  general semi-definite positive matrix, 
                            'fixed': fixed matrix specified in fixed_trait_covar,
                            'diag': diagonal matrix, 
                            'lowrank': low rank matrix. The rank of the lowrank part is specified in the variable rank,
                            'lowrank_id': sum of a lowrank matrix and a multiple of the identity. The rank of the lowrank part is specified in the variable rank, 
                            'lowrank_diag': sum of a lowrank and a diagonal matrix. The rank of the lowrank part is specified in the variable rank, 
                            'block': multiple of a matrix of ones,
                            'block_id': sum of a multiple of a matrix of ones and a multiple of the idenity,
                            'block_diag': sum of a multiple of a matrix of ones and a diagonal matrix,
            rank:       rank of a possible lowrank component (default 1)
            fixed_trait_covar:   PxP matrix for the (predefined) trait-to-trait covariance matrix if fixed type is used
        """

        assert K is not None or is_noise, 'VarianceDecomposition:: Specify covariance structure'
        
        if is_noise:
            assert self.noisPos is None, 'VarianceDecomposition:: Noise term already exists'
            K  = sp.eye(self.N)
            Kcross = None
            self.noisPos = self.n_randEffs
        else:
            assert K.shape[0]==self.N, 'VarianceDecomposition:: Incompatible shape for K'
            assert K.shape[1]==self.N, 'VarianceDecomposition:: Incompatible shape for K'
        if Kcross is not None:
            assert self.Ntest is not None, 'VarianceDecomposition:: specify Ntest for predictions (method VarianceDecomposition::setTestSampleSize)'
            assert Kcross.shape[0]==self.N, 'VarianceDecomposition:: Incompatible shape for Kcross'
            if self.Ntest is None: self.Ntest = Kcross.shape[1]
            else:
                assert Kcross.shape[1]==self.Ntest, 'VarianceDecomposition:: Incompatible shape for Kcross'
 
        if normalize:
            Norm = 1/K.diagonal().mean()
            K *= Norm
            if Kcross is not None: Kcross *= Norm

        _C = self._buildTraitCovar(trait_covar_type=trait_covar_type,rank=rank,fixed_trait_covar=fixed_trait_covar)
        self.C.append(_C)
        self.R.append(K)
        self.Rstar.append(Kcross)
        self.n_randEffs+=1
    
        self.gp         = None
        self.init       = False
        self.fast       = False

        self.cache['Sigma']   = None
        self.cache['Hessian'] = None

    def addFixedEffect(self,F=None,A=None,Ftest=None):
        """
        add fixed effect term to the model

        Args:
            F:     sample design matrix for the fixed effect [N,K]
            A:     trait design matrix for the fixed effect (e.g. sp.ones((1,P)) common effect; sp.eye(P) any effect) [L,P]
            Ftest: sample design matrix for test samples [Ntest,K]
        """
        if A is None:
            A = sp.eye(self.P)
        if F is None:
            F = sp.ones((self.N,1))
            if self.Ntest is not None:
                Ftest = sp.ones((self.Ntest,1)) 
        
        assert A.shape[1]==self.P, 'VarianceDecomposition:: A has incompatible shape'
        assert F.shape[0]==self.N, 'VarianceDecimposition:: F has incompatible shape'
       
        # Ftest should also go in mu for predicting
        self.mu.addFixedEffect(F=F,A=A)

        self.gp      = None
        self.init    = False
        self.fast    = False

        self.cache['Sigma']   = None
        self.cache['Hessian'] = None

    def trainGP(self,fast=None):
        """
        Train the gp
       
        Args:
            fast:       if true and the gp has not been initialized, initializes a kronSum gp
            scales0:    initial variance components params
        """
        assert self.n_randEffs>0, 'VarianceDecomposition:: No variance component terms'

        if not self.init:        self._initGP(fast=fast)

        # LIMIX CVARIANCEDECOMPOSITION TRAINING
        params0 = self.gp.getParams()
        conv,info = OPT.opt_hyper(self.gp,params0,factr=1e3)
        
        self.cache['Sigma']   = None
        self.cache['Hessian'] = None
            
        return conv

    def optimize(self,fast=None,scales0=None,fixed0=None,init_method=None,termx=0,n_times=10,perturb=True,pertSize=1e-3,verbose=None):
        """
        Train the model using the specified initialization strategy
        
        Args:
            fast:            if true, fast gp is considered; if None (default), fast inference is considered if possible
            scales0:        if not None init_method is set to manual
            fixed0:         initial weights for fixed effects
            init_method:    initialization strategy:
                                'random': variance component parameters (scales) are sampled from a normal distribution with mean 0 and std 1,
                                'diagonal': uses the a two-random-effect single trait model to initialize the parameters,
                                'manual': the starting point is set manually,
            termx:            term used for initialization in the diagonal strategy
            n_times:        number of restarts to converge
            perturb:        if true, the initial point (set manually opr through the single-trait model) is perturbed with gaussian noise
            perturbSize:    std of the gassian noise used to perturb the initial point
            verbose:        print if convergence is achieved and how many restarted were needed 
        """
        if not self.init:        self._initGP(fast=fast)

        if init_method is None:         init_method = 'empirical'
        if scales0 is not None:         init_method = 'manual'
        if init_method is 'random':     perturb = False
        if init_method in ['diagonal','manual','empirical']:
            if not perturb:        n_times = 1
        
        if init_method=='empirical':
            Yres = self.mu.getResiduals()
            cov = sp.cov(Yres.T)/float(self.n_randEffs)
            for term_i in range(self.n_randEffs):
                self.C[term_i].setCovariance(cov)

        for i in range(n_times):
            if init_method=='random':
                for term_i in range(self.n_randEffs):
                    self.C[term_i].setRandomParams()
            elif perturb:
                for term_i in range(self.n_randEffs):
                    self.C[term_i].perturbParams(pertSize)
            conv = self.trainGP()
            if conv:    break
    
        if verbose:
            if conv==False:
                print 'No local minimum found for the tested initialization points'
            else:
                print 'Local minimum found at iteration %d' % i
 
        return conv

    def getLML(self):
        """
        Return log marginal likelihood
        """
        assert self.init, 'GP not initialised'
        return self.gp.LML()


    def getLMLgrad(self):
        """
        Return gradient of log-marginal likelihood
        """
        assert self.init, 'GP not initialised'
        return self.gp.LMLgrad()


    def setScales(self,scales=None,term_num=None):
        """
        set Cholesky parameters (scales)

        Args:
            scales:     values of Cholesky parameters
                        if scales is None, sample them randomly from a gaussian distribution with mean 0 and std 1,
            term_num:   index of the term whose variance paraments are to be set
                        if None, scales is interpreted as the stack vector of the variance parameters from all terms
        """
        if scales is None:
            for term_i in range(self.n_randEffs):
                n_scales = self.C[term_i].getParams().shape(0)
                self.C[term_i].setParams(sp.randn(n_scales))
        elif term_num is None:
            n_scales = 0
            for term_i in range(self.n_randEffs):
                n_scales += self.C[term_i].getParams().shape(0)
            assert scales.shape[0]==n_scales, 'incompatible shape'
            index = 0
            for term_i in range(self.n_randEffs):
                index1 = index+self.vd.getTerm(term_i).getNumberScales()
                C[term_i].setParams(scales[index:index1])
                index = index1
        else:
            assert scales.shape[0]==self.vd.getTerm(term_num).getNumberScales(), 'incompatible shape'
            self.vd.getTerm(term_num).setScales(scales)

    def getScales(self,term_i=None):
        """
        Returns Cholesky parameters
        To retrieve proper variances and covariances \see getVarComps and \see getTraitCovar
        
        Args:
            term_i:     index of the term of which we want to retrieve the variance paramenters
        Returns:
            vector of cholesky parameters of term_i
            if term_i is None, the stack vector of the cholesky parameters from all terms is returned
        """
        if term_i is None:
            RV = []
            for term_i in range(self.n_randEffs):
                RV.append(self.C[term_i].getParams())
            RV = sp.concatenate(RV)
        else:
            assert term_i<self.n_randEffs, 'Term index non valid'
            RV = self.C[term_i].getParams()
        return RV


    def getWeights(self):
        """
        Return stack vector of the weight of all fixed effects
        """
        print "DEPRECATED"
        assert self.init, 'GP not initialised'
        return self.gp.getParams()['dataTerm']


    def getTraitCovar(self,term_i):
        """
        Return the estimated trait covariance matrix for term_i (or the total if term_i is None)

        Args:
            term_i:     index of the random effect term we want to retrieve the covariance matrix
        Returns:
            estimated trait correlation coefficie
        """
        assert term_i<self.n_randEffs, 'Term index non valid'
        return self.C[term_i].K()


    def getVarianceComps(self,univariance=False):
        """
        Return the estimated variance components

        Args:
            univariance:   Boolean indicator, if True variance components are normalized to sum up to 1 for each trait
        Returns:
            variance components of all random effects on all phenotypes [P, n_randEffs matrix]
        """
        RV=sp.zeros((self.P,self.n_randEffs))
        for term_i in range(self.n_randEffs):
            RV[:,term_i] = self.C[term_i].K().diagonal()
        if univariance:
            RV /= RV.sum(1)[:,sp.newaxis]
        return RV


    """ standard errors """

    def getTraitCovarStdErrors(self,term_i):
        """
        Returns standard errors on trait covariances from term_i (for the covariance estimate \see getTraitCovar)

        Args:
            term_i:     index of the term we are interested in
        """
        print 'DEPRECATED'
        assert self.init,        'GP not initialised'
        assert self.fast==False, 'Not supported for fast implementation'
        
        if self.P==1:
            out = (2*self.getScales()[term_i])**2*self._getLaplaceCovar()[term_i,term_i]
        else:
            C = self.vd.getTerm(term_i).getTraitCovar()
            n_params = C.getNumberParams()
            par_index = 0
            for term in range(term_i-1):
                par_index += self.vd.getTerm(term_i).getNumberScales()
            Sigma1 = self._getLaplaceCovar()[par_index:(par_index+n_params),:][:,par_index:(par_index+n_params)]
            out = sp.zeros((self.P,self.P))
            for param_i in range(n_params):
                out += C.Kgrad_param(param_i)**2*Sigma1[param_i,param_i]
                for param_j in range(param_i):
                    out += 2*abs(C.Kgrad_param(param_i)*C.Kgrad_param(param_j))*Sigma1[param_i,param_j]
        out = sp.sqrt(out)
        return out

    def getVarianceCompStdErrors(self,univariance=False):
        """
        Return the standard errors on the estimated variance components (for variance component estimates \see getVarianceComps)

        Args:
            univariance:   Boolean indicator, if True variance components are normalized to sum up to 1 for each trait
        Returns:
            standard errors on variance components [P, n_randEffs matrix]
        """
        print 'DEPRECATED'
        RV=sp.zeros((self.P,self.n_randEffs))
        for term_i in range(self.n_randEffs):
            RV[:,term_i] = self.getTraitCovarStdErrors(term_i).diagonal()
        var = getVarianceComps()
        if univariance:
            RV /= var.sum(1)[:,sp.newaxis]
        return RV

    """
    CODE FOR PREDICTIONS
    """
    
    def predictPhenos(self,use_fixed=None,use_random=None):
        """
        predict the conditional mean (BLUP)

        Args:
            use_fixed:        list of fixed effect indeces to use for predictions
            use_random:        list of random effect indeces to use for predictions
        Returns:
            predictions (BLUP)
        """
        print 'DEPRECATED'
        assert self.noisPos is not None,      'No noise element'
        assert self.init,               'GP not initialised'
        assert self.Ntest is not None,        'VarianceDecomposition:: specify Ntest for predictions (method VarianceDecomposition::setTestSampleSize)'
 
        use_fixed  = range(self.n_fixedEffs)
        use_random = range(self.n_randEffs)

        KiY = self.gp.agetKEffInvYCache()
                
        if self.fast==False:
            KiY = KiY.reshape(self.P,self.N).T
                
        Ypred = sp.zeros((self.Ntest,self.P))

        # predicting from random effects
        for term_i in use_random:
            if term_i!=self.noisPos:
                Kstar = self.Kstar[term_i]
                if Kstar is None:
                    warnings.warn('warning: random effect term %d not used for predictions as it has None cross covariance'%term_i)
                    continue 
                term  = sp.dot(Kstar.T,KiY)
                if self.P>1:
                    C    = self.getTraitCovar(term_i)
                    term = sp.dot(term,C)
                else:
                    term *= self.getVarianceComps()[0,term_i]
                Ypred += term

        # predicting from fixed effects
        weights = self.getWeights()
        w_i = 0
        for term_i in use_fixed:
            Fstar = self.Fstar[term_i]
            if Fstar is None:
                warnings.warn('warning: fixed effect term %d not used for predictions as it has None test sample design'%term_i)
                continue 
            if self.P==1:    A = sp.eye(1)
            else:            A = self.vd.getDesign(term_i)
            Fstar = self.Fstar[term_i]
            W = weights[w_i:w_i+A.shape[0],0:1].T 
            term = sp.dot(Fstar,sp.dot(W,A))
            w_i += A.shape[0]
            Ypred += term

        return Ypred


    def crossValidation(self,seed=0,n_folds=10,fullVector=True,verbose=None,D=None,**keywords):
        """
        Split the dataset in n folds, predict each fold after training the model on all the others

        Args:
            seed:        seed
            n_folds:     number of folds to train the model on
            fullVector:  Bolean indicator, if true it stops if no convergence is observed for one of the folds, otherwise goes through and returns a pheno matrix with missing values
            verbose:     if true, prints the fold that is being used for predicitons
            **keywords:  params to pass to the function optimize
        Returns:
            Matrix of phenotype predictions [N,P]
        """
        print 'DEPRECATED'
        # split samples into training and test
        sp.random.seed(seed)
        r = sp.random.permutation(self.Y.shape[0])
        nfolds = 10
        Icv = sp.floor(((sp.ones((self.Y.shape[0]))*nfolds)*r)/self.Y.shape[0])

        RV = {}
        if self.P==1:  RV['var'] = sp.zeros((nfolds,self.n_randEffs))
        else:          RV['var'] = sp.zeros((nfolds,self.P,self.n_randEffs))

        Ystar = sp.zeros_like(self.Y)

        for fold_j in range(n_folds):

            if verbose:
                print ".. predict fold %d"%fold_j

            Itrain  = Icv!=fold_j
            Itest   = Icv==fold_j
            Ytrain  = self.Y[Itrain,:]
            Ytest   = self.Y[Itest,:]
            vc = VarianceDecomposition(Ytrain)
            vc.setTestSampleSize(Itest.sum())
            for term_i in range(self.n_fixedEffs):
                F      = self.vd.getFixed(term_i)
                Ftest  = F[Itest,:]
                Ftrain = F[Itrain,:]
                if self.P>1:    A = self.vd.getDesign(term_i)
                else:           A = None
                vc.addFixedEffect(F=Ftrain,Ftest=Ftest,A=A)
            for term_i in range(self.n_randEffs):
                if self.P>1:
                    tct  = self.trait_covar_type[term_i]
                    rank = self.rank[term_i]
                    ftc  = self.fixed_tc[term_i]
                    jitt = self.jitter[term_i]
                    if tct=='lowrank_diag1' or tct=='freeform1':
                        d = D[fold_j,:,term_i]
                    else:
                        d = None
                else:
                    tct  = None
                    rank = None
                    ftc  = None
                    jitt = None
                    d    = None
                if term_i==self.noisPos:
                    vc.addRandomEffect(is_noise=True,trait_covar_type=tct,rank=rank,jitter=jitt,fixed_trait_covar=ftc,d=d)
                else:
                    R = self.vd.getTerm(term_i).getK()
                    Rtrain = R[Itrain,:][:,Itrain]
                    Rcross = R[Itrain,:][:,Itest]
                    vc.addRandomEffect(K=Rtrain,Kcross=Rcross,trait_covar_type=tct,rank=rank,jitter=jitt,fixed_trait_covar=ftc,d=d)
            conv = vc.optimize(verbose=False,**keywords)
            if self.P==1:
                RV['var'][fold_j,:] = vc.getVarianceComps()[0,:]
            else:
                RV['var'][fold_j,:,:] = vc.getVarianceComps()

            if fullVector:
                assert conv, 'VarianceDecompositon:: not converged for fold %d. Stopped here' % fold_j
            if conv: 
                Ystar[Itest,:] = vc.predictPhenos()
            else:
                warnings.warn('not converged for fold %d' % fold_j)
                Ystar[Itest,:] = sp.nan

        return Ystar,RV


    """ GP initialization """

    def _initGP(self,fast=None):
        """
        Initialize GP objetct
        
        Args:
            fast:   Boolean indicator denoting if fast implementation is to consider
                    if fast is None (default) and possible in the specifc situation, fast implementation is considered
        """
        if fast is None:        fast = (self.n_randEffs==2) and (~sp.isnan(self.Y).any())
        elif fast:
            assert self.n_randEffs==2, 'VarianceDecomposition: for fast inference number of random effect terms must be == 2'
            assert not sp.isnan(self.Y).any(), 'VarianceDecomposition: fast inference available only for complete phenotype designs'
        if fast:
            self.gp = limix.core.gp.gp2kronSum(self.mu,
                        Cg=self.C[1-self.noisPos],
                        Cn=self.C[self.noisPos],
                        XX=self.R[1-self.noisPos])
        else:
            # TODO
            pass
        self.init=True
        self.fast=fast


    def _buildTraitCovar(self,trait_covar_type='lowrank_diag',rank=1,fixed_trait_covar=None,jitter=1e-4,d=None):
        """
        Internal functions that builds the trait covariance matrix using the LIMIX framework

        Args:
            trait_covar_type: type of covaraince to use. Default 'freeform'. possible values are 
            rank:       rank of a possible lowrank component (default 1)
            fixed_trait_covar:   PxP matrix for the (predefined) trait-to-trait covariance matrix if fixed type is used
        Returns:
            LIMIX::PCovarianceFunction for Trait covariance matrix
            vector labelling Cholesky parameters for different initializations
        """
        if trait_covar_type=='freeform':
            cov = limix.core.covar.freeform(self.P)
        else:
            assert True==False, 'VarianceDecomposition:: trait_covar_type not valid'
        return cov
        



    """ INTERNAL FUNCIONS FOR PARAMETER INITIALIZATION """

    def _getH2singleTrait(self, K, verbose=None):
        """
        Internal function for parameter initialization
        estimate variance components and fixed effect using a linear mixed model with an intercept and 2 random effects (one is noise)
        Args:
            K:        covariance matrix of the non-noise random effect term 
        """
        # Fit single trait model
        varg  = sp.zeros(self.P)
        varn  = sp.zeros(self.P)
        fixed = sp.zeros((1,self.P))
        
        for p in range(self.P):
            y = self.Y[:,p:p+1]
            # check if some sull value
            I  = sp.isnan(y[:,0])
            if I.sum()>0:
                y  = y[~I,:]
                _K = K[~I,:][:,~I]
            else:
                _K  = copy.copy(K)
            lmm = limix.CLMM()
            lmm.setK(_K)
            lmm.setSNPs(sp.ones((y.shape[0],1)))
            lmm.setPheno(y)
            lmm.setCovs(sp.zeros((y.shape[0],1)))
            lmm.setVarcompApprox0(-20, 20, 1000)
            lmm.process()
            delta = sp.exp(lmm.getLdelta0()[0,0])
            Vtot  = sp.exp(lmm.getLSigma()[0,0])
    
            varg[p] = Vtot
            varn[p] = delta*Vtot
            fixed[:,p] = lmm.getBetaSNP()
    
            if verbose: print p
    
        sth = {}
        sth['varg']  = varg
        sth['varn']  = varn
        sth['fixed'] = fixed

        return sth

    def _getScalesDiag(self,termx=0):
        """
        Internal function for parameter initialization
        Uses 2 term single trait model to get covar params for initialization
        
        Args:
            termx:      non-noise term terms that is used for initialization 
        """
        print 'DEPRECATED'
        assert self.P>1, 'VarianceDecomposition:: diagonal init_method allowed only for multi trait models' 
        assert self.noisPos is not None, 'VarianceDecomposition:: noise term has to be set'
        assert termx<self.n_randEffs-1, 'VarianceDecomposition:: termx>=n_randEffs-1'
        assert self.trait_covar_type[self.noisPos] not in ['lowrank','block','fixed'], 'VarianceDecomposition:: diagonal initializaiton not posible for such a parametrization'
        assert self.trait_covar_type[termx] not in ['lowrank','block','fixed'], 'VarianceDecimposition:: diagonal initializaiton not posible for such a parametrization'
        scales = []
        res = self._getH2singleTrait(self.vd.getTerm(termx).getK())
        scaleg = sp.sqrt(res['varg'].mean())
        scalen = sp.sqrt(res['varn'].mean())
        for term_i in range(self.n_randEffs):
            if term_i==termx:
                _scales = scaleg*self.diag[term_i]
            elif term_i==self.noisPos:
                _scales = scalen*self.diag[term_i]
            else:
                _scales = 0.*self.diag[term_i]
            if self.jitter[term_i]>0:
                _scales = sp.concatenate((_scales,sp.array([sp.sqrt(self.jitter[term_i])])))
            scales.append(_scales)
        return sp.concatenate(scales)

    """ INTERNAL FUNCIONS FOR ESTIMATING PARAMETERS UNCERTAINTY """

    def _getHessian(self):
        """
        Internal function for estimating parameter uncertainty 
        COMPUTES OF HESSIAN OF E(\theta) = - log L(\theta | X, y)
        """
        assert self.init,        'GP not initialised'
        assert self.fast==False, 'Not supported for fast implementation'
        
        if self.cache['Hessian'] is None:
            ParamMask=self.gp.getParamMask()['covar']
            std=sp.zeros(ParamMask.sum())
            H=self.gp.LMLhess_covar()
            It= (ParamMask[:,0]==1)
            self.cache['Hessian']=H[It,:][:,It]
        
        return self.cache['Hessian']


    def _getLaplaceCovar(self):
        """
        Internal function for estimating parameter uncertainty 
        Returns:
            the 
        """
        assert self.init,        'GP not initialised'
        assert self.fast==False, 'Not supported for fast implementation'
        
        if self.cache['Sigma'] is None:
            self.cache['Sigma'] = sp.linalg.inv(self._getHessian())
        return self.cache['Sigma']

    def _getFisher(self):
        """
        Return the fisher information matrix over the parameters
        TO CHECK: IT DOES NOT WORK PROPERLY
        """
        Ctot = self.vd.getGP().getCovar()
        Ki = sp.linalg.inv(Ctot.K())
        n_scales = self.vd.getNumberScales()
        out = sp.zeros((n_scales,n_scales))
        for m in range(n_scales):
            out[m,m] = 0.5 * sp.trace(sp.dot(Ki,sp.dot(Ctot.Kgrad_param(m),sp.dot(Ki,Ctot.Kgrad_param(m)))))
            for n in range(m):
                out[m,n]=0.5 * sp.trace(sp.dot(Ki,sp.dot(Ctot.Kgrad_param(m),sp.dot(Ki,Ctot.Kgrad_param(n)))))
                out[n,m]=out[m,n]
        return out

    """ DEPRECATED """

    def _getModelPosterior(self,min):
        """
        USES LAPLACE APPROXIMATION TO CALCULATE THE BAYESIAN MODEL POSTERIOR
        """
        Sigma = self._getLaplaceCovar(min)
        n_params = self.vd.getNumberScales()
        ModCompl = 0.5*n_params*sp.log(2*sp.pi)+0.5*sp.log(sp.linalg.det(Sigma))
        RV = min['LML']+ModCompl
        return RV

