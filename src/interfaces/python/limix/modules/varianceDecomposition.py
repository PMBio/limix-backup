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
        initialise:                  Initialise parameters randomly
        fit:                         Fit phenos and returns the minimum with some info
        fit_ntimes:                  Fit phenos ntimes with different random initialization and returns the list of minima found in the order of LML
        getParams:                   Returns the stack vector of the parameters of the trait covariances C_i
        getEmpTraitCovar:            Returns the empirical trait covariance
        getEstTraitCovar:            Returns the total trait covariance in the GP by summing up the trait covariances C_i
        getCovParams:                Calculates the inverse hessian of the -loglikelihood with respect to the parameters (covariance matrix of the posterior over params under laplace approximation)
        estimateHeritabilities:      Given K, it fits the model with 1 global fixed effects and covariance matrix hg[p]*K+hn[p] for each trait p and returns the vectors hg and hn
    """
    
    
    def __init__(self,Y,standardize=True):
        """
        Y: phenotype matrix [N, P ]
        F: fixed effect
        K: list of intra-trait covariance matrix
        """
        
        #create column of 1 for fixed if nothing providede
        self.N       = Y.shape[0]
        self.P       = Y.shape[1]
        self.Nt      = self.N*self.P
        
        if standardize==True:
            Y = SP.stats.zscore(Y)
        
        self.Y       = Y
        self.vd      = limix.CNewVarianceDecomposition(Y)
        self.n_terms = 0
        self.gp      = None
        
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
        self.vd.addTerm('Diagonal',K)
    
    def addFixedTerm(self,F):
        """
        set the fixed effect term
        F: fixed effect matrix [N,P]
        """
        assert F.shape[0]==self.N, 'Incompatible shape'
        assert F.shape[1]==self.P, 'Incompatible shape'
        
        self.vd.addFixedEffTerm(F)
    
    def initialise(self):
        """
        get random initialization of variances based on the empirical trait variance
        """
        for term_i in range(self.n_terms):
            n_params = self.vd.getTerm(term_i).getNumberScales()
            self.vd.getTerm(term_i).setScales(SP.array(SP.randn(n_params)))
        """
        I HAVE TO FIGURE THIS OUT
        """
        """
        EmpVarY=self.getEmpTraitVar()
        if self.P==1:
            temp=SP.rand(self.n_terms)
            N=temp.sum()
            temp=temp/N*EmpVarY
            self.vd.setScales(temp)
        else:
            temp=SP.rand(2*self.n_terms,self.P)
            N=temp.sum(0)
            temp=temp/N*EmpVarY
            for term_i in range(self.n_terms):
                a=(2*SP.rand(self.P)-1)*SP.sqrt(temp[term_i,:])
                c=SP.sqrt(temp[term_i,:])
                C0 = SP.dot(a[:,SP.newaxis],a[:,SP.newaxis].T)+SP.diag(c**2)
        """
    
    def fit(self):
        """
        fit a variance component model with the predefined design and the initialization and returns all the results
        """
        
        assert self.n_terms>0, 'No variance component terms'
        
        Params0 = self.getParams()
        
        # GP initialisation
        self.vd.initGP()
        gp =self.vd.getGP()
        LML0=-1.0*gp.LML()
    
        # LIMIX CVARIANCEDECOMPOSITION FITTING
        start_time = time.time()
        conv =self.vd.trainGP()
        time_train=time.time()-start_time
        
        # Check whether limix::CVarianceDecomposition.train() has converged
        LMLgrad = self.vd.getLMLgrad()
        if conv!=True:
            print 'limix::CVarianceDecomposition::train has not converged'
            res=None
        else:
            res = {
                'Params0':          Params0,
                'Params':           self.getParams(),
                'LML':              SP.array([-1.0*gp.LML()]),
                'LML0':             SP.array([LML0]),
                'LMLgrad':          SP.array([LMLgrad]),
                'time_train':       SP.array([time_train]),
                'gp' :              gp
                }
        return res
        pass
    
    
    def fit_ntimes(self,ntimes=10,dist_mins=1e-2):
        """
        fit phenos ntimes with different random initialization and returns the minima order with respect to gp.LML
        """

        optima=[]
        LML=SP.zeros((1,0))
        
        for i in range(ntimes):
            
            print ".. Minimization %d" % i
            
            self.initialise()
            min=self.fit()
        
            if min!=None:
                temp=1
                for j in range(len(optima)):
                    if SP.linalg.norm(min['Params']-optima[j]['Params'])<dist_mins:
                        temp=0
                        optima[j]['counter']+=1
                        break
                if temp==1:
                    min['counter']=1
                    optima.append(min)
                    LML=SP.concatenate((LML,min['LML'][:,SP.newaxis]),1)
    
         # Order the list optima with respect to LML; the first optimum has highest LML
        optima1=[]
        index = LML.argsort(axis = 1)[0,:][::-1]
        for i in range(len(optima)):
            optima1.append(optima[index[i]])
    
        return optima1

    def getParams(self,term_i=None):
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

    def getEstTraitCovar(self,term_i=None):
        """
        Returns the estimated trait covariance matrix
        term_i: index of the term we are interested in
        """
        if term_i==None:
            RV=SP.zeros((self.P,self.P))
            for term_i in range(self.n_terms): RV+=self.vd.getTraitCovariance(term_i)
        else:
            assert term_i<self.n_terms, 'Term index non valid'
            RV = self.vd.getTraitCovariance(term_i)
        return RV

    def getCovParams(self,min):
        """
        USES LAPLACE APPROXIMATION TO CALCULATE THE COVARIANCE MATRIX OF THE OPTIMIZED PARAMETERS
        """
        gp=min['gp']
        ParamMask=gp.getParamMask()['covar']
        std=SP.zeros(ParamMask.sum())
        H=gp.LMLhess(["covar"])
        It= (ParamMask[:,0]==1)
        H=H[It,:][:,It]
        Sigma = SP.linalg.inv(H)
        return Sigma
    
    def getModelPosterior(self,min,Sigma=None):
        """
        USES LAPLACE APPROXIMATION TO CALCULATE THE BAYESIAN MODEL POSTERIOR
        """
        
        if Sigma==None:
            Sigma = self.getCovParams(min)
        
        n_params = 0
        for term_i in range(self.n_terms):
            n_params += self.C[term_i].getNumberParams()
        
        ModCompl = 0.5*n_params*SP.log(2*SP.pi)+0.5*SP.log(SP.linalg.det(Sigma))
        
        RV = min['LML']+ModCompl
            
        return RV

    def getEmpTraitCovar(self):
        """
        Returns the empirical trait covariance matrix
        """
        return SP.cov(self.Y.T)
    
    def getEmpTraitVar(self):
        """
        Returns the vector of empirical trait variances
        """
        if self.P==1:
            RV = self.getEmpTraitCovar()
        else:
            RV = self.getEmpTraitCovar().diagonal()
        return RV

    def estimateHeritabilities(self,K):
        """
        It fits the model with 1 fixed effects and covariance matrix h1[p]*K+h2[p] for each trait p and return the vectors h1 and h2
        """
    
        hg = SP.zeros(self.P)
        hn = SP.zeros(self.P)
    
        for p in range(self.P):
            hg[p], hn[p]=limix.CVarianceDecomposition.aestimateHeritability(self.Y[:,p:p+1],SP.ones(self.N),K)
    
        return hg, hn



