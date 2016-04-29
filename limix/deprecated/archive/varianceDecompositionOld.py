import sys

import scipy as SP
import scipy.linalg
import scipy.stats
import limix
import limix.utils.preprocess as preprocess
import pdb
import time
import copy

class VarianceDecomposition:
    """
    Variance decomposition module in LIMIX
    This class mainly takes care of initialization and eases the interpretation of complex variance decompositions

    #TODO: change documentation. Python modules explain what the functions do but a list of function is not needed, I think?
    #A 3 line example how to use it would be good to have here... perhaps.
    Methods:
        __init__(self,Y):            Constructor
        addSingleTraitTerm:          Add Single Trait Term
        addMultiTraitTerm:           Add Multi Trait Term (form of trait-to-trait covariance matrix need to be set as well)
        addFixedTerm:                Add Fixed Effect Term
        fit:                         Fit phenos and returns the minimum with some info
        setScales:                   Set the vector of covar parameterss
        getScales:                   Return the vector of the covar parameters
        getFixed:                    Return the vector of the dataTerm parameters
        getEmpTraitCovar:            Return the empirical trait covariance
        getEstTraitCovar:            Return the estimated trait-to-trait covariance matrix for term i
        estimateHeritabilities:      Given K, it fits the model with 1 global fixed effects and covariance matrix hg[p]*K+hn[p] for each trait p and returns the vectors hg and hn
    """
    
    
    def __init__(self,Y,standardize=False):
        """
        Args:
            Y:              phenotype matrix [N, P]
            standardize:    if True, impute missing phenotype values by mean value,
                            zero-mean and unit-variance phenotype (Boolean, default False)
        """
        
        #create column of 1 for fixed if nothing providede
        self.N       = Y.shape[0]
        self.P       = Y.shape[1]
        self.Nt      = self.N*self.P
        self.Iok     = ~(SP.isnan(Y).any(axis=1))

        #outsourced to handle missing values:
        if standardize:
            Y=preprocess.standardize(Y)

        self.Y       = Y
        self.vd      = limix.CVarianceDecomposition(Y)
        self.n_terms = 0
        self.gp      = None
        self.init    = False
        self.fast    = False
        self.noisPos = None
        self.optimum = None

        # for multi trait models 
        self.covar_type = []
        self.diag       = []
        self.offset     = []
        
        self.cache = {}
        self.cache['Sigma']   = None
        self.cache['Hessian'] = None
        self.cache['Lparams'] = None
        self.cache['paramsST']= None

        pass
    
    
    def setY(self,Y,standardize=False):
        """
        Set phenotype matrix
        
        Args:
            Y:              phenotype matrix [N, P]
            standardize:	if True, phenotype is standardized (zero mean, unit variance)
        """
        assert Y.shape[0]==self.N, 'CVarianceDecomposition:: Incompatible shape'
        assert Y.shape[1]==self.P, 'CVarianceDecomposition:: Incompatible shape'

        
        if standardize:
            Y=preprocess.standardize(Y)

        #check that missing values match the current structure
        assert (~(SP.isnan(Y).any(axis=1))==self.Iok).all(), 'CVarianceDecomposition:: pattern of missing values needs to match Y given at initialization'

        self.Y = Y
        self.vd.setPheno(Y)

        self.optimum = None
    
        self.cache['Sigma']   = None
        self.cache['Hessian'] = None
        self.cache['Lparams'] = None
        self.cache['paramsST']= None

    def addRandomEffect(self,K=None,covar_type='freeform',is_noise=False,normalize=True,Ks=None,offset=1e-4,rank=1,covar_K0=None):
        """
        Add random effect Term
        depending on self.P=1 or >1 add single trait or multi trait random effect term
        """
        if self.P==1:	self.addSingleTraitTerm(K=K,is_noise=is_noise,normalize=normalize,Ks=Ks)
        else:			self.addMultiTraitTerm(K=K,covar_type=covar_type,is_noise=is_noise,normalize=normalize,Ks=Ks,offset=offset,rank=rank,covar_K0=covar_K0)

    
    def addSingleTraitTerm(self,K=None,is_noise=False,normalize=True,Ks=None):
        """
        add random effects term for single trait models (no trait-trait covariance matrix)
        
        Args:
            K:          NxN sample covariance matrix
            is_noise:	bool labeling the noise term (noise term has K=eye)
            normalize:	if True, K and Ks are scales such that K.diagonal().mean()==1
            Ks:			NxN test cross covariance for predictions
        """
        
        assert self.P == 1, 'Incompatible number of traits'
        
        assert K!=None or is_noise, 'Specify covariance structure'
        
        if is_noise:
            assert self.noisPos==None, 'noise term already exists'
            K = SP.eye(self.Nt)
            self.noisPos = self.n_terms
        else:
            assert K.shape[0]==self.Nt, 'Incompatible shape'
            assert K.shape[1]==self.Nt, 'Incompatible shape'
    
        if Ks!=None:
            assert Ks.shape[0]==self.N, 'Incompatible shape'

        if normalize:
            Norm = 1/K.diagonal().mean()
            K *= Norm
            if Ks!=None: Ks *= Norm

        self.vd.addTerm(limix.CSingleTraitTerm(K))
        if Ks!=None: self.setKstar(self.n_terms,Ks)
        self.n_terms+=1
    
        self.gp         = None
        self.init       = False
        self.fast       = False
        self.optimum    = None

        self.cache['Sigma']   = None
        self.cache['Hessian'] = None
        self.cache['Lparams'] = None
        self.cache['paramsST']= None

    
    #TODO: rename offset. Perhaps we can call it regularization or whiteNoise or something. 
    def addMultiTraitTerm(self,K=None,covar_type='freeform',is_noise=False,normalize=True,Ks=None,offset=1e-4,rank=1,covar_K0=None):
        """
        add multi trait random effects term.
        The inter-trait covariance is parametrized by covar_type, where parameters are optimized.
        
        Args:
            K:      Individual-individual (Intra-Trait) Covariance Matrix [N, N]
                    (K is normalised in the C++ code such that K.trace()=N)
            covar_type: type of covaraince to use. Default 'freeform'. possible values are 
                            'freeform': free form optimization, 
                            'fixed': use a fixed matrix specified in covar_K0,
                            'diag': optimize a diagonal matrix, 
                            'lowrank': optimize a low rank matrix. The rank of the lowrank part is specified in the variable rank,
                            'lowrank_id': optimize a low rank matrix plus the weight of a constant diagonal matrix. The rank of the lowrank part is specified in the variable rank, 
                            'lowrank_diag': optimize a low rank matrix plus a free diagonal matrix. The rank of the lowrank part is specified in the variable rank, 
                            'block': optimize the weight of a constant P x P block matrix of ones,
                            'block_id': optimize the weight of a constant P x P block matrix of ones plus the weight of a constant diagonal matrix,
                            'block_diag': optimize the weight of a constant P x P block matrix of ones plus a free diagonal matrix,                            
            is_noise:   Boolean indicator specifying if the matrix is homoscedastic noise (weighted identity covariance) (default False)
            normalize:  Boolean indicator specifying if K is normalized such that K.trace()=N.
            Ks:			NxNtest cross covariance for predictions
            offset:		diagonal contribution added to trait-to-trait covariance matrices for regularization
            rank:       rank of a possible lowrank component (default 1)
            covar_K0:   PxP matrix for the (predefined) trait-to-trait covariance matrix if fixed type is used
        """
        assert self.P > 1, 'CVarianceDecomposition:: Incompatible number of traits'
        assert K!=None or is_noise, 'CVarianceDecomposition:: Specify covariance structure'
        assert offset>=0, 'CVarianceDecomposition:: offset must be >=0'

        #TODO: check that covar_K0 is correct if fixed typeCF is used..

        if is_noise:
            assert self.noisPos==None, 'CVarianceDecomposition:: noise term already exists'
            K = SP.eye(self.N)
            self.noisPos = self.n_terms
        else:
            assert K.shape[0]==self.N, 'CVarianceDecomposition:: Incompatible shape'
            assert K.shape[1]==self.N, 'CVarianceDecomposition:: Incompatible shape'

        if Ks!=None:
            assert Ks.shape[0]==self.N, 'CVarianceDecomposition:: Incompatible shape'

        if normalize:
            Norm = 1/K.diagonal().mean()
            K *= Norm
            if Ks!=None: Ks *= Norm

        cov = limix.CSumCF()
        if covar_type=='freeform':
            cov.addCovariance(limix.CFreeFormCF(self.P))
            L = SP.eye(self.P)
            diag = SP.concatenate([L[i,:(i+1)] for i in range(self.P)])
        elif covar_type=='fixed':
            cov.addCovariance(limix.CFixedCF(covar_K0))
            diag = SP.zeros(1)
        elif covar_type=='diag':
            cov.addCovariance(limix.CDiagonalCF(self.P))
            diag = SP.ones(self.P)
        elif covar_type=='lowrank':
            cov.addCovariance(limix.CLowRankCF(self.P,rank))
            diag = SP.zeros(self.P*rank)
        elif covar_type=='lowrank_id':
            cov.addCovariance(limix.CLowRankCF(self.P,rank))
            cov.addCovariance(limix.CFixedCF(SP.eye(self.P)))
            diag = SP.concatenate([SP.zeros(self.P*rank),SP.ones(1)])
        elif covar_type=='lowrank_diag':
            cov.addCovariance(limix.CLowRankCF(self.P,rank))
            cov.addCovariance(limix.CDiagonalCF(self.P))
            diag = SP.concatenate([SP.zeros(self.P*rank),SP.ones(self.P)])
        elif covar_type=='block':
            cov.addCovariance(limix.CFixedCF(SP.ones((self.P,self.P))))
            diag = SP.zeros(1)
        elif covar_type=='block_id':
            cov.addCovariance(limix.CFixedCF(SP.ones((self.P,self.P))))
            cov.addCovariance(limix.CFixedCF(SP.eye(self.P)))
            diag = SP.concatenate([SP.zeros(1),SP.ones(1)])
        elif covar_type=='block_diag':
            cov.addCovariance(limix.CFixedCF(SP.ones((self.P,self.P))))
            cov.addCovariance(limix.CDiagonalCF(self.P))
            diag = SP.concatenate([SP.zeros(1),SP.ones(self.P)])
        else:
            assert True==False, 'CVarianceDecomposition:: covar_type not valid'

        if offset>0:
            _cov = limix.CFixedCF(SP.eye(self.P))
            _cov.setParams(SP.array([SP.sqrt(offset)]))
            _cov.setParamMask(SP.zeros(1))
            cov.addCovariance(_cov)
        self.offset.append(offset)

        self.covar_type.append(covar_type)
        self.diag.append(diag)

        self.vd.addTerm(cov,K)
        if Ks!=None: self.setKstar(self.n_terms,Ks)
        self.n_terms+=1

        self.gp      = None
        self.init    = False
        self.fast    = False
        self.optimum = None

        self.cache['Sigma']   = None
        self.cache['Hessian'] = None
        self.cache['Lparams'] = None
        self.cache['paramsST']= None
    
    
    def addFixedEffect(self,F=None,A=None):
        """
        add fixed effect to the model

        Args:
            F: fixed effect matrix [N,1]
            A: design matrix [K,P] (e.g. SP.ones((1,P)) common effect; SP.eye(P) any effect)
        """
        if A==None:
            A = SP.eye(self.P)
        if F==None:
            F = SP.ones((self.N,1))
        
        assert A.shape[1]==self.P, 'Incompatible shape'
        assert F.shape[0]==self.N, 'Incompatible shape'
       
        if F.shape[1]>1:
            for m in range(F.shape[1]):
                self.vd.addFixedEffTerm(A,F[:,m:m+1])
        else:
            self.vd.addFixedEffTerm(A,F)

        #TODO: what is this gp object doing, is this initialization correct?
        self.gp      = None
        self.init    = False
        self.fast    = False
        self.optimum = None

        self.cache['Sigma']   = None
        self.cache['Hessian'] = None
        self.cache['Lparams'] = None
        self.cache['paramsST']= None

    #TODO: automatically select the fast inference method if applicable. 
    #We can create a method "isFast()" to check whether the model is compatible with the fast version but really this can be automated.
    def initGP(self,fast=False):
        """
        Initialize GP objetct
        
        Args:
            fast:   if fast==True initialize gpkronSum gp
        """
        if fast:
            assert self.n_terms==2, 'CVarianceDecomposition: for fast inference number of terms must be == 2'
            assert self.P>1,        'CVarianceDecomposition: for fast inference number of traits must be > 1'
            self.vd.initGPkronSum()
        else:
            self.vd.initGP()
        self.gp=self.vd.getGP()
        self.init=True
        self.fast=fast

    #TODO: move all internal function to the end and flag them as such
    def _getScalesDiag(self,termx=0):
        """
        Uses 2 term single trait model to get covar params for initialization
        
        Args:
            termx:      non-noise term terms that is used for initialization 
        """
        assert self.P>1, 'CVarianceDecomposition:: diagonal init_method allowed only for multi trait models' 
        assert self.noisPos!=None, 'CVarianceDecomposition:: noise term has to be set'
        assert termx<self.n_terms-1, 'CVarianceDecomposition:: termx>=n_terms-1'
        assert self.covar_type[self.noisPos] not in ['lowrank','block','fixed'], 'CVarianceDecimposition:: diagonal initializaiton not posible for such a parametrization'
        assert self.covar_type[termx] not in ['lowrank','block','fixed'], 'CVarianceDecimposition:: diagonal initializaiton not posible for such a parametrization'
        scales = []
        res = self.estimateHeritabilities(self.vd.getTerm(termx).getK())
        scaleg = SP.sqrt(res['varg'].mean())
        scalen = SP.sqrt(res['varn'].mean())
        for term_i in range(self.n_terms):
            if term_i==termx:
                _scales = scaleg*self.diag[term_i]
            elif term_i==self.noisPos:
                _scales = scalen*self.diag[term_i]
            else:
                _scales = 0.*self.diag[term_i]
            if self.offset[term_i]>0:
                _scales = SP.concatenate((_scales,SP.array([SP.sqrt(self.offset[term_i])])))
            scales.append(_scales)
        return SP.concatenate(scales)

    def _getScalesRand(self):
        """
        Return a vector of random scales
        """
        if self.P>1:
            scales = []
            for term_i in range(self.n_terms):
                _scales = SP.randn(self.diag[term_i].shape[0])
                if self.offset[term_i]>0:
                    _scales = SP.concatenate((_scales,SP.array([SP.sqrt(self.offset[term_i])])))
                scales.append(_scales)
            scales = SP.concatenate(scales)
        else:
            scales=SP.randn(self.vd.getNumberScales())

        return scales

    def _perturbation(self):
        """
        Returns Gaussian perturbation
        """
        if self.P>1:
            scales = []
            for term_i in range(self.n_terms):
                _scales = SP.randn(self.diag[term_i].shape[0])
                if self.offset[term_i]>0:
                    _scales  = SP.concatenate((_scales,SP.zeros(1)))
                scales.append(_scales)
            scales = SP.concatenate(scales)
        else:
            scales = SP.randn(self.vd.getNumberScales())
        return scales
 
    def trainGP(self,fast=False,scales0=None,fixed0=None,lambd=None):
        """
        Train the gp
       
        Args:
            fast:       if true and the gp has not been initialized, initializes a kronSum gp
            scales0:	initial variance components params
            fixed0:     initial fixed effect params
        """
        assert self.n_terms>0, 'CVarianceDecomposition:: No variance component terms'

        if not self.init:		self.initGP(fast=fast)

        # set lambda
        if lambd!=None:		self.gp.setLambda(lambd)

        # set scales0
        if scales0!=None:
            self.setScales(scales0)
        # init gp params
        self.vd.initGPparams()
        # set fixed0
        if fixed0!=None:
            params = self.gp.getParams()
            params['dataTerm'] = fixed0
            self.gp.setParams(params)

        # LIMIX CVARIANCEDECOMPOSITION TRAINING
        conv =self.vd.trainGP()
        
        self.cache['Sigma']   = None
        self.cache['Hessian'] = None
            
        return conv

    #TOOD: auto select of fast (see initGP)
    #Q: is initGP not an internal method? Is the user going to call this himself? 
    def findLocalOptimum(self,fast=False,scales0=None,fixed0=None,init_method=None,termx=0,n_times=10,perturb=True,pertSize=1e-3,verbose=True,lambd=None):
        """
        Train the model using the specified initialization strategy
        
        Args:
            fast:		    if true, fast gp is initialized
            scales0:        if not None init_method is set to manual
            fixed0:         initial fixed effects
            init_method:    initialization method \in {random,diagonal,manual} 
            termx:			term for diagonal diagonalisation
            n_times:        number of times the initialization
            perturb:        if true, the initial point is perturbed with gaussian noise
            perturbSize:    size of the perturbation
            verbose:        print if convergence is achieved
        """

        if init_method==None:
            if self.P==1:	init_method = 'random'
            else:           init_method = 'diagonal'

        if not self.init:		self.initGP(fast=fast)

        if scales0!=None and ~perturb: 	init_method = 'manual'
        
        if init_method=='diagonal':
            scales0 = self._getScalesDiag(termx=termx)
            
        if init_method=='diagonal' or init_method=='manual':
            if not perturb:		n_times = 1

        if fixed0==None:
            fixed0 = SP.zeros_like(self.gp.getParams()['dataTerm'])

        for i in range(n_times):
            if init_method=='random':
                scales1 = self._getScalesRand()
                fixed1  = pertSize*SP.randn(fixed0.shape[0],fixed0.shape[1])
            elif perturb:
                scales1 = scales0+pertSize*self._perturbation()
                fixed1  = fixed0+pertSize*SP.randn(fixed0.shape[0],fixed0.shape[1])
            else:
                scales1 = scales0
                fixed1  = fixed0
            conv = self.trainGP(scales0=scales1,fixed0=fixed1,lambd=lambd)
            if conv:    break
    
        if verbose:
            if conv==False:
                print('No local minimum found for the tested initialization points')
            else:
                print(('Local minimum found at iteration %d' % i))
 
        return conv

    #TODO: this function and the above seem redundant. 
    #Don't you want this function to wrap the above, i.e. write the above as internal function and have this as the autside interface?
    #having multiple restarts is a simple add on. You can pass arguments automatically using the **kw_args keyword, which avoids having to redefine options here that are used again in the
    #internal function
    def findLocalOptima(self,fast=False,verbose=True,n_times=10,lambd=None):
        """
        Train the model repeadly up to a number specified by the users with random restarts and
        return a list of all relative minima that have been found 
        
        Args:
            fast:       Boolean. if set to True initalize kronSumGP
            verbose:    Boolean. If set to True, verbose output is produced. (default True)
            n_times:    number of re-starts of the optimization. (default 10)
        """
        if not self.init:       self.initGP(fast)
        
        opt_list = []

        fixed0 = SP.zeros_like(self.gp.getParams()['dataTerm'])    

        # minimises n_times
        for i in range(n_times):
            
            scales1 = self._getScalesRand()
            fixed1  = 1e-1*SP.randn(fixed0.shape[0],fixed0.shape[1])
            conv = self.trainGP(fast=fast,scales0=scales1,fixed0=fixed1,lambd=lambd)

            if conv:
                # compare with previous minima
                temp=1
                for j in range(len(opt_list)):
                    if SP.allclose(abs(self.getScales()),abs(opt_list[j]['scales'])):
                        temp=0
                        opt_list[j]['counter']+=1
                        break
                if temp==1:
                    opt = {}
                    opt['counter'] = 1
                    opt['LML'] = self.getLML()
                    opt['scales'] = self.getScales()
                    opt_list.append(opt)
        
        
        # sort by LML
        LML = SP.array([opt_list[i]['LML'] for i in range(len(opt_list))])
        index   = LML.argsort()[::-1]
        out = []
        if verbose:
            print("\nLocal mimima\n")
            print("n_times\t\tLML")
            print("------------------------------------")
            for i in range(len(opt_list)):
                out.append(opt_list[index[i]])
                if verbose:
                    print(("%d\t\t%f" % (opt_list[index[i]]['counter'], opt_list[index[i]]['LML'])))
                print("")

        return out

    #TODO: Q: externally visible function?
    def getLML(self):
        """
        Return log marginal likelihood
        """
        assert self.init, 'GP not initialised'
        return self.vd.getLML()


    #TODO: Q: externally visible function?
    def getLMLgrad(self):
        """
        Return gradient of log-marginal likelihood
        """
        assert self.init, 'GP not initialised'
        return self.vd.getLMLgrad()

    #TODO: text not clear get random ....
    def setScales(self,scales=None,term_num=None):
        """
        get random initialization of variances based on the empirical trait variance

        Args:
            scales:     if scales==None: set them randomly, 
                        else: set scales to term_num (if term_num==None: set to all terms)
            term_num:   set scales to term_num
        """
        if scales==None:
            for term_i in range(self.n_terms):
                n_scales = self.vd.getTerm(term_i).getNumberScales()
                self.vd.getTerm(term_i).setScales(SP.array(SP.randn(n_scales)))
        elif term_num==None:
            assert scales.shape[0]==self.vd.getNumberScales(), 'incompatible shape'
            index = 0
            for term_i in range(self.n_terms):
                index1 = index+self.vd.getTerm(term_i).getNumberScales()
                self.vd.getTerm(term_i).setScales(scales[index:index1])
                index = index1
        else:
            assert scales.shape[0]==self.vd.getTerm(term_num).getNumberScales(), 'incompatible shape'
            self.vd.getTerm(term_num).setScales(scales)

    #TODO: needs more text
    def getScales(self,term_i=None):
        """
        Returns the Parameters
        
        Args:
            term_i:     index of the term we are interested in
                        if term_i==None returns the whole vector of parameters
        """
        if term_i==None:
            RV = self.vd.getScales()
        else:
            assert term_i<self.n_terms, 'Term index non valid'
            RV = self.vd.getScales(term_i)
        return RV


    #TODO: naming? We use addFixedEffect. and getFixed. 
    def getFixed(self):
        """
        Return dataTerm params
        """
        assert self.init, 'GP not initialised'
        return self.gp.getParams()['dataTerm']


    def getEstTraitCovar(self,term_i=None):
        """
        Returns explicitly the estimated trait covariance matrix

        Args:
            term_i:     index of the term we are interested in
        """
        assert self.P>1, 'Trait covars not defined for single trait analysis'
        
        if term_i==None:
            RV=SP.zeros((self.P,self.P))
            for term_i in range(self.n_terms): RV+=self.vd.getTerm(term_i).getTraitCovar().K()
        else:
            assert term_i<self.n_terms, 'Term index non valid'
            RV = self.vd.getTerm(term_i).getTraitCovar().K()
        return RV


    #TODO: naming? suggests that this calculates one correlation coefficient rather than a matrix.
    #Add references between getEstTraitCorrCoeff and getEstTraitCovar
    def getEstTraitCorrCoef(self,term_i=None):
        """
        Returns the estimated trait correlation matrix

        Args:
            term_i:     index of the term we are interested in
        """
        cov = self.getEstTraitCovar(term_i)
        stds=SP.sqrt(cov.diagonal())[:,SP.newaxis]
        RV = cov/stds/stds.T
        return RV
    

    #TODO: naming? How does this relate to getScales?
    def getVariances(self):
        """
        Returns the estimated variances as a n_terms x P matrix
        each row of the output represents a term and its P values represent the variance corresponding variance in each trait
        """
        if self.P>1:
            RV=SP.zeros((self.n_terms,self.P))
            for term_i in range(self.n_terms):
                RV[term_i,:]=self.vd.getTerm(term_i).getTraitCovar().K().diagonal()
        else:
            RV=self.getScales()**2
        return RV


    #TODO: naming? How does this relate to getScales and getVariances?
    def getVarComponents(self):
        """
        Returns the estimated variance components as a n_terms x P matrix
        each row of the output represents a term and its P values represent the variance component corresponding variance in each trait
        """
        RV = self.getVariances()
        RV /= RV.sum(0)
        return RV

    #TODO: externally visible function? THe external interface needs to be as static as possible. We should minimize this.
    def getHessian(self):
        """
        COMPUTES OF HESSIAN OF E(\theta) = - log L(\theta | X, y)
        """
        assert self.init,        'GP not initialised'
        assert self.fast==False, 'Not supported for fast implementation'
        
        if self.cache['Hessian']==None:
            ParamMask=self.gp.getParamMask()['covar']
            std=SP.zeros(ParamMask.sum())
            H=self.gp.LMLhess_covar()
            It= (ParamMask[:,0]==1)
            self.cache['Hessian']=H[It,:][:,It]
        
        return self.cache['Hessian']
    
    #TODO: externally visible function? THe external interface needs to be as static as possible. We should minimize this.
    def getLaplaceCovar(self):
        """
        USES LAPLACE APPROXIMATION TO CALCULATE THE COVARIANCE MATRIX OF THE OPTIMIZED PARAMETERS
        """
        assert self.init,        'GP not initialised'
        assert self.fast==False, 'Not supported for fast implementation'
        
        if self.cache['Sigma']==None:
            self.cache['Sigma'] = SP.linalg.inv(self.getHessian())
        return self.cache['Sigma']

    #TODO: externally visible function? THe external interface needs to be as static as possible. We should minimize this.
    def getFisher(self):
        """
        Return the fisher information matrix over the parameters
        TO CHECK: IT DOES NOT WORK PROPERLY
        """
        Ctot = self.vd.getGP().getCovar()
        Ki = SP.linalg.inv(Ctot.K())
        n_scales = self.vd.getNumberScales()
        out = SP.zeros((n_scales,n_scales))
        for m in range(n_scales):
            out[m,m] = 0.5 * SP.trace(SP.dot(Ki,SP.dot(Ctot.Kgrad_param(m),SP.dot(Ki,Ctot.Kgrad_param(m)))))
            for n in range(m):
                out[m,n]=0.5 * SP.trace(SP.dot(Ki,SP.dot(Ctot.Kgrad_param(m),SP.dot(Ki,Ctot.Kgrad_param(n)))))
                out[n,m]=out[m,n]
        return out
    
    #TODO: naming. not clear what the standard errors refer to. I want getVarianceParams, getVarainceParamsSTtd() or similar.
    #one could also consider having std calculation as argument in the getVarianceParams.
    def getStdErrors(self,term_i):
        """
        RETURNS THE STANDARD DEVIATIONS ON VARIANCES AND CORRELATIONS BY PROPRAGATING THE UNCERTAINTY ON LAMBDAS

        Args:
            term_i:     index of the term we are interested in
        """
        assert self.init,        'GP not initialised'
        assert self.fast==False, 'Not supported for fast implementation'
        
        if self.P==1:
            out = (2*self.getScales()[term_i])**2*self.getLaplaceCovar()[term_i,term_i]
        else:
            C = self.vd.getTerm(term_i).getTraitCovar()
            n_params = C.getNumberParams()
            par_index = 0
            for term in range(term_i-1):
                par_index += self.vd.getTerm(term_i).getNumberScales()
            Sigma1 = self.getLaplaceCovar()[par_index:(par_index+n_params),:][:,par_index:(par_index+n_params)]
            out = SP.zeros((self.P,self.P))
            for param_i in range(n_params):
                out += C.Kgrad_param(param_i)**2*Sigma1[param_i,param_i]
                for param_j in range(param_i):
                    out += 2*abs(C.Kgrad_param(param_i)*C.Kgrad_param(param_j))*Sigma1[param_i,param_j]
        out = SP.sqrt(out)
        return out
    
    #TODO: externally visible function? THe external interface needs to be as static as possible. We should minimize this.
    def getModelPosterior(self,min):
        """
        USES LAPLACE APPROXIMATION TO CALCULATE THE BAYESIAN MODEL POSTERIOR
        """
        Sigma = self.getLaplaceCovar(min)
        n_params = self.vd.getNumberScales()
        ModCompl = 0.5*n_params*SP.log(2*SP.pi)+0.5*SP.log(SP.linalg.det(Sigma))
        RV = min['LML']+ModCompl
        return RV


    #TODO: externally visible function? THe external interface needs to be as static as possible. We should minimize this.
    def getEmpTraitCovar(self):
        """
        Returns the empirical trait covariance matrix
        """
        if self.P==1:
            out=self.Y[self.Iok].var()
        else:
            out=SP.cov(self.Y[self.Iok].T)
        return out


    #TODO: externally visible function? THe external interface needs to be as static as possible. We should minimize this.
    def getEmpTraitCorrCoef(self):
        """
        Returns the empirical trait correlation matrix
        """
        cov = self.getEmpTraitCovar()
        stds=SP.sqrt(cov.diagonal())[:,SP.newaxis]
        RV = cov/stds/stds.T
        return RV
    
    
    #TODO: externally visible function? THe external interface needs to be as static as possible. We should minimize this.
    def estimateHeritabilities(self, K, verbose=False):
        """
        estimate variance components and fixed effects
        from a single trait model having only two terms
        """
        
        # Fit single trait model
        varg  = SP.zeros(self.P)
        varn  = SP.zeros(self.P)
        fixed = SP.zeros((1,self.P))
        
        for p in range(self.P):
            y = self.Y[:,p:p+1]
            lmm = limix.CLMM()
            lmm.setK(K)
            lmm.setSNPs(SP.ones((K.shape[0],1)))
            lmm.setPheno(y)
            lmm.setCovs(SP.zeros((K.shape[0],1)))
            lmm.setVarcompApprox0(-20, 20, 1000)
            lmm.process()
            delta = SP.exp(lmm.getLdelta0()[0,0])
            Vtot  = SP.exp(lmm.getLSigma()[0,0])
    
            varg[p] = Vtot
            varn[p] = delta*Vtot
            fixed[:,p] = lmm.getBetaSNP()
    
            if verbose: print(p)
    
        sth = {}
        sth['varg']  = varg
        sth['varn']  = varn
        sth['fixed'] = fixed

        return sth

    """
    CODE FOR PREDICTIONS
    """
    
    #TODO: naming. not nice... needs also quite a few more checks, I think.
    def setKstar(self,term_i,Ks):
        """
        Set the kernel for predictions

        Args:
            term_i:     index of the term we are interested in
            Ks:         (TODO: is this the covariance between train and test or the covariance between test points?)
        """
        assert Ks.shape[0]==self.N
    
        #if Kss!=None:
            #assert Kss.shape[0]==Ks.shape[1]
            #assert Kss.shape[1]==Ks.shape[1]

        self.vd.getTerm(term_i).getKcf().setK0cross(Ks)


    #TODO: naming. not nice... needs also quite a few more checks, I think.
    def predictMean(self):
        """
        predict the conditional mean (BLUP)

        Returns:
            predictions (BLUP)
        """
        assert self.noisPos!=None,      'No noise element'
        assert self.init,               'GP not initialised'
                
        KiY = self.gp.agetKEffInvYCache()
                
        if self.fast==False:
            KiY = KiY.reshape(self.P,self.N).T
                
        Ymean = None
        for term_i in range(self.n_terms):
            if term_i!=self.noisPos:
                Kstar = self.vd.getTerm(term_i).getKcf().Kcross(SP.zeros(1))
                term  = SP.dot(Kstar.T,KiY)
                if self.P>1:
                    C    = self.getEstTraitCovar(term_i)
                    term = SP.dot(term,C)
                if Ymean==None:     Ymean  = term
                else:               Ymean += term
            
        # to generalise
        Ymean+=self.getFixed()[:,0]
                
        return Ymean

