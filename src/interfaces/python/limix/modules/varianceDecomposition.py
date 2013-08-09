import sys
sys.path.append('./..')
sys.path.append('./../../..')

import scipy as SP
import scipy.linalg
import scipy.stats
import limix
import pdb
import time



class CVarianceDecomposition:
    """
    helper function for variance decomposition in limix
    This class mainly takes care of initialization and interpreation of results
    """
    
    """
    Methods:
        __init__(self,Y):            Constructor
        addSingleTraitTerm:          Add Single Trait Term
        addMultiTraitTerm:           Add Multi Trait Term (inter-trait Covariance Matrix is FreeForm)
        addFixedTerm:                Add Fixed Effect Term
        fit:                         Fit phenos and returns the minimum with some info
        setScales:                   Set the vector of covar parameterss
        getScales:                   Return the vector of the covar parameters
        getFixed:                    Return the vector of the dataTerm parameters
        getEmpTraitCovar:            Return the empirical trait covariance
        getEstTraitCovar:            Return the total trait covariance in the GP by summing up the trait covariances C_i
        getLaplaceCovar:             Calculate the inverse hessian of the -loglikelihood with respect to the parameters (covariance matrix of the posterior over params under laplace approximation)
        estimateHeritabilities:      Given K, it fits the model with 1 global fixed effects and covariance matrix hg[p]*K+hn[p] for each trait p and returns the vectors hg and hn
    """
    
    
    def __init__(self,Y,standardize=True):
        """
        Y: phenotype matrix [N, P ]
        """
        
        #create column of 1 for fixed if nothing providede
        self.N       = Y.shape[0]
        self.P       = Y.shape[1]
        self.Nt      = self.N*self.P
        
        if standardize==True:
            Y = SP.stats.zscore(Y)
        
        self.Y       = Y
        self.vd      = limix.CVarianceDecomposition(Y)
        self.n_terms = 0
        self.gp      = None
        self.init    = False
        self.fast    = False
        self.optimum = None
        
        pass

    def getCore(self):
        return self.vd
    
    def getNumberTerms(self):
        return self.n_terms
    
    def addSingleTraitTerm(self,K):
        """
        add single trait term
        K:  Intra-Trait Covariance Matrix [N, N]
        (K is normalised in the C++ code such that K.trace()==N)
        """
        assert self.P == 1, 'Incompatible number of traits'
        
        assert K.shape[0]==self.Nt, 'Incompatible shape'
        assert K.shape[1]==self.Nt, 'Incompatible shape'

        self.n_terms+=1
        self.vd.addTerm(limix.CSingleTraitTerm(K))
    
    def addMultiTraitTerm(self,K):
        """
        add multi trait term (inter-trait covariance matrix is consiered freeform)
        K:  Intra-Trait Covariance Matrix [N, N]
        (K is normalised in the C++ code such that K.trace()==N)
        """
        assert self.P > 1, 'Incompatible number of traits'
        
        assert K.shape[0]==self.N, 'Incompatible shape'
        assert K.shape[1]==self.N, 'Incompatible shape'
        
        self.n_terms+=1
        self.vd.addTerm(limix.CFreeFormCF(self.P),K)
    
    def addFixedTerm(self,F,A=None):
        """
        set the fixed effect term
        A: design matrix [K,P] (e.g. SP.ones((1,P)) common effect; SP.eye(P) specific effect))
        F: fixed effect matrix [N,1]
        """
        if A==None:
            A = SP.eye(self.P)
        
        assert A.shape[0]==self.P, 'Incompatible shape'
        assert F.shape[0]==self.N, 'Incompatible shape'
        assert F.shape[1]==1,      'Incompatible shape'
        
        self.vd.addFixedEffTerm(A,F)
    
    def fit(self,fast=False):
        """
        fit a variance component model with the predefined design and the initialization and returns all the results
        if fast=True, use GPkronSum (valid only if P>1 and n_terms=1) 
        """
        
        assert self.n_terms>0, 'No variance component terms'
        
        params0 = self.getScales()
        
        # GP initialisation
        if fast:
            assert self.n_terms==2, 'error: number of terms must be 2'
            assert self.P>1,        'error: number of traits must be > 1'
            self.vd.initGPkronSum()
        else:
            self.vd.initGP()
        self.gp =self.vd.getGP()
        LML0=-1.0*self.gp.LML()
            
        # LIMIX CVARIANCEDECOMPOSITION FITTING
        start_time = time.time()
        conv =self.vd.trainGP()
        time_train=time.time()-start_time
        
        # Check whether limix::CVarianceDecomposition.train() has converged
        LMLgrad = self.vd.getLMLgrad()
        if conv:
            self.optimum = {
                'params0':          params0,
                'params':           self.getScales(),
                'LML':              SP.array([-1.0*self.gp.LML()]),
                'LML0':             SP.array([LML0]),
                'LMLgrad':          SP.array([LMLgrad]),
                'time_train':       SP.array([time_train]),
            }
        
        else:   print 'limix::CVarianceDecomposition::train has not converged'
            
        self.init = True
        self.fast = fast
            
        return conv
        pass


    def getOptimum(self):
        return self.optimum

    def getLML(self):
        """
        Return LML
        """
        assert self.init, 'GP not initialised'
        return self.vd.getLML()
    
    def getLMLgrad(self):
        """
        Return LMLgrad
        """
        assert self.init, 'GP not initialised'
        return self.vd.getLMLgrad()

    def setScales(self,scales=None,term_num=None):
        """
        get random initialization of variances based on the empirical trait variance
        if scales==None:    set them randomly
        else:               set scales to term_i (if term_i==None: set to all terms)
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

    
    def getScales(self,term_i=None):
        """
        Returns the Parameters
        term_i: index of the term we are interested in
        if term_i==None returns the whole vector of parameters
        """
        if term_i==None:
            RV = self.vd.getScales()
        else:
            assert term_i<self.n_terms, 'Term index non valid'
            RV = self.vd.getScales(term_i)
        return RV

    def getFixed(self):
        """
        Return dataTerm params
        """
        assert self.init, 'GP not initialised'
        return self.gp.getParams()['dataTerm']

    def getEstTraitCovar(self,term_i=None):
        """
        Returns the estimated trait covariance matrix
        term_i: index of the term we are interested in
        """
        assert self.P>1, 'Trait covars not defined for single trait analysis'
        
        if term_i==None:
            RV=SP.zeros((self.P,self.P))
            for term_i in range(self.n_terms): RV+=self.vd.getTerm(term_i).getTraitCovar().K()
        else:
            assert term_i<self.n_terms, 'Term index non valid'
            RV = self.vd.getTerm(term_i).getTraitCovar().K()
        return RV
    
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
    
    def getVarComponents(self):
        """
            Returns the estimated variance components as a n_terms x P matrix
            each row of the output represents a term and its P values represent the variance component corresponding variance in each trait
        """
        RV = self.getVariances()
        RV /= RV.sum(0)
        return RV

    def getLaplaceCovar(self):
        """
        USES LAPLACE APPROXIMATION TO CALCULATE THE COVARIANCE MATRIX OF THE OPTIMIZED PARAMETERS
        """
        assert self.init,        'GP not initialised'
        assert self.fast==False, 'Not supported for fast implementation'
        
        ParamMask=self.gp.getParamMask()['covar']
        std=SP.zeros(ParamMask.sum())
        H=self.gp.LMLhess_covar()
        It= (ParamMask[:,0]==1)
        H=H[It,:][:,It]
        Sigma = SP.linalg.inv(H)
        return Sigma
    
    def getModelPosterior(self,min,Sigma=None):
        """
        USES LAPLACE APPROXIMATION TO CALCULATE THE BAYESIAN MODEL POSTERIOR
        """
        if Sigma==None:
            Sigma = self.getLaplaceCovar(min)
        n_params = self.vd.getNumberScales()
        ModCompl = 0.5*n_params*SP.log(2*SP.pi)+0.5*SP.log(SP.linalg.det(Sigma))
        RV = min['LML']+ModCompl
        return RV

    def getEmpTraitCovar(self):
        """
        Returns the empirical trait covariance matrix
        """
        return SP.cov(self.Y.T)

    def estimateHeritabilities(self,K):
        """
        It fits the model with 1 fixed effects and covariance matrix h1[p]*K+h2[p] for each trait p and return the vectors h1 and h2
        """
    
        hg = SP.zeros(self.P)
        hn = SP.zeros(self.P)
    
        for p in range(self.P):
            hg[p], hn[p]=limix.CVarianceDecomposition.aestimateHeritability(self.Y[:,p:p+1],SP.ones(self.N),K)
    
        return hg, hn



