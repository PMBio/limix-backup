import sys
sys.path.append('./..')
sys.path.append('./../../..')

import scipy as SP
import scipy.linalg
import scipy.stats
import limix
import limix.utils.preprocess as preprocess
import pdb
import time
import copy

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
    
    
    def __init__(self,Y,standardize=False):
        """
        Y: phenotype matrix [N, P ]
        """
        
        #create column of 1 for fixed if nothing providede
        self.N       = Y.shape[0]
        self.P       = Y.shape[1]
        self.Nt      = self.N*self.P
        self.Iok     = ~(SP.isnan(Y).any(axis=1))

        #outsourced to handle missing values:
        if standardize:
            preprocess.standardize(Y,in_place=True)

        self.Y       = Y
        self.vd      = limix.CVarianceDecomposition(Y)
        self.n_terms = 0
        self.gp      = None
        self.init    = False
        self.fast    = False
        self.noisPos = None
        self.optimum = None
        
        self.cache = {}
        self.cache['Sigma']   = None
        self.cache['Hessian'] = None
        self.cache['Lparams'] = None
        self.cache['paramsST']= None

        pass
    
    
    def setY(self,Y,standardize=False):
        
        assert Y.shape[0]==self.N, 'Incompatible shape'
        assert Y.shape[1]==self.P, 'Incompatible shape'
        
        if standardize:
            preprocess.standardize(Y,in_place=True)

        #check that missing values match the current structure
        assert (~(SP.isnan(Y).any(axis=1))==self.Iok).all(), 'pattern of missing values needs to match Y given at initialization'

        self.Y = Y
        self.vd.setPheno(Y)

        self.optimum = None
    
        self.cache['Sigma']   = None
        self.cache['Hessian'] = None
        self.cache['Lparams'] = None
        self.cache['paramsST']= None

    
    def addSingleTraitTerm(self,K=None,is_noise=False,normalize=True,Ks=None):
        """
        add single trait term
        is_noise:   if is_noise: I->K
        K:          Intra-Trait Covariance Matrix [N, N]
        (K is normalised in the C++ code such that K.trace()==N)
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
    
        self.gp      = None
        self.init    = False
        self.fast    = False
        self.optimum = None

        self.cache['Sigma']   = None
        self.cache['Hessian'] = None
        self.cache['Lparams'] = None
        self.cache['paramsST']= None

    
    
    def addMultiTraitTerm(self,K=None,covar_type='freeform',covar_K0=None,is_noise=False,normalize=True,Ks=None):
        """
        add multi trait term (inter-trait covariance matrix is consiered freeform)
        K:  Intra-Trait Covariance Matrix [N, N]
        (K is normalised in the C++ code such that K.trace()==N)
        covar_type: type of covaraince to use. Default "freeform"
        covar_K0: fixed CF covariance (if covar_type=='fixed')
        """
        assert self.P > 1, 'Incompatible number of traits'
        
        assert K!=None or is_noise, 'Specify covariance structure'
        
        if is_noise:
            assert self.noisPos==None, 'noise term already exists'
            K = SP.eye(self.N)
            self.noisPos = self.n_terms
        else:
            assert K.shape[0]==self.N, 'Incompatible shape'
            assert K.shape[1]==self.N, 'Incompatible shape'

        if Ks!=None:
            assert Ks.shape[0]==self.N, 'Incompatible shape'

        if normalize:
            Norm = 1/K.diagonal().mean()
            K *= Norm
            if Ks!=None: Ks *= Norm

        if covar_type=='freeform':
            cov = limix.CFreeFormCF(self.P)
        elif covar_type=='fixed':
            cov = limix.CFixedCF(covar_K0)
        elif covar_type=='rank1':
            cov = limix.CRankOneCF(self.P)
        elif covar_type=='diag':
            cov = limix.CDiagonalCF(self.P)
        elif covar_type=='rank1_diag':
            cov=limix.CSumCF()
            cov.addCovariance(limix.CRankOneCF(self.P))
            if self.P==2:
                cov.addCovariance(limix.CFixedCF(SP.eye(self.P)))
            else:
                cov.addCovariance(limix.CDiagonalCF(self.P))

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
    
        self.gp      = None
        self.init    = False
        self.fast    = False
        self.optimum = None

        self.cache['Sigma']   = None
        self.cache['Hessian'] = None
        self.cache['Lparams'] = None
        self.cache['paramsST']= None
    
            
    def fit(self,fast=False,init_method='empCov'):
        """
        fit a variance component model with the predefined design and the initialization and returns all the results
        if fast=True, use GPkronSum (valid only if P>1 and n_terms==2) 
        """
        
        assert self.n_terms>0, 'No variance component terms'
        
        # Parameter initialisation
        assert init_method in ['manual','empCov','singleTrait','random','mixed'], 'method %s not known' % init_method
        
        if init_method=='mixed':
            method_box = ['empCov','random']
            if self.n_terms==2 and self.P>1:    method_box.append('singleTrait')
            method = SP.random.permutation(method_box)[0]
        else:
            method = init_method
        
        if method == 'random':
            self.setScales()
        elif method == 'empCov':
            self.initEmpirical()
            self.perturbScales()
        elif method == 'singleTrait':
            self.initSingleTrait()
            self.perturbScales()
        

        params0 = self.getScales()
        
        # GP initialisation
        if fast:
            assert self.n_terms==2, 'error: number of terms must be 2'
            assert self.P>1,        'error: number of traits must be > 1'
            self.vd.initGPkronSum()
        else:
            self.vd.initGP()
        self.gp=self.vd.getGP()
                
        LML0=-1.0*self.gp.LML()

        # set Fixed effect randomly
        params = self.gp.getParams()
        params['dataTerm'] = 1e-2*SP.randn(params['dataTerm'].shape[0],params['dataTerm'].shape[1])

        # LIMIX CVARIANCEDECOMPOSITION FITTING
        start_time = time.time()
        conv =self.vd.trainGP()
        time_train=time.time()-start_time
        
        # Check whether limix::CVarianceDecomposition.train() has converged
        LMLgrad = self.vd.getLMLgrad()
        if conv:
            self.optimum = {
                'params0':          params0,
                'LML0':             LML0,
                'init_method':      method,
                'time_train':       SP.array([time_train]),
                
                'LML':              -1.0*self.gp.LML(),
                'params':           self.getScales()
            }
            
        self.init    = True
        self.fast    = fast

        self.cache['Sigma']   = None
        self.cache['Hessian'] = None
            
        return conv
        pass

    
    def findLocalOptimum(self,fast=False,init_method='empCov',verbose=True,n_times=10):
        """
        Train the model up to n_times using the random method initilisation method init_method
        and stop as soon as a local optimum is found
        """
        
        for i in range(n_times):
            conv = self.fit(fast=fast,init_method=init_method)
            if conv:    break
    
        if verbose:
            if conv==False:
                print 'No local minimum found for the tested initialization points'
            else:
                print 'Local minimum found at iteration %d' % i
                
        return conv
            
            
    def findLocalOptima(self,fast=False,init_method='empCov',verbose=True,n_times=10):
        """
        Train the model n_times using the random method initilisation method init_method
        and returns a list of the local optima that have been found
        """
        
        opt_list = []
    
        # minimises n_times
        for i in range(n_times):
            
            if self.fit(fast=fast,init_method=init_method):
                # checks whether the minimum was found before
                temp=1
                for j in range(len(opt_list)):
                    if (abs(self.optimum['params'])-abs(opt_list[j]['params'])).max()<1e-6:
                        temp=0
                        opt_list[j]['counter']+=1
                        break
                if temp==1:
                    optimum = self.optimum
                    optimum['counter'] = 1
                    opt_list.append(optimum)
        
        if verbose: print "n_times\t\tLML"
        
        out = []
                        
        # sort by LML
        LML = SP.array([opt_list[i]['LML'] for i in range(len(opt_list))])
        index   = LML.argsort()[::-1]
        out = []
        for i in range(len(opt_list)):
            out.append(opt_list[index[i]])
            if verbose:
                print "%d\t\t%f" % (opt_list[index[i]]['counter'], opt_list[index[i]]['LML'])

        return out
    


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


    def getEstTraitCorrCoef(self,term_i=None):
        """
        Returns the estimated trait correlation matrix
        term_i: index of the term we are interested in
        """
        cov = self.getEstTraitCovar(term_i)
        stds=SP.sqrt(cov.diagonal())[:,SP.newaxis]
        RV = cov/stds/stds.T
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
    
    def getLaplaceCovar(self):
        """
        USES LAPLACE APPROXIMATION TO CALCULATE THE COVARIANCE MATRIX OF THE OPTIMIZED PARAMETERS
        """
        assert self.init,        'GP not initialised'
        assert self.fast==False, 'Not supported for fast implementation'
        
        if self.cache['Sigma']==None:
            self.cache['Sigma'] = SP.linalg.inv(self.getHessian())
        return self.cache['Sigma']

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
    
    def getStdErrors(self,term_i):
        """
        RETURNS THE STANDARD DEVIATIONS ON VARIANCES AND CORRELATIONS BY PROPRAGATING THE UNCERTAINTY ON LAMBDAS
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
    
    def getModelPosterior(self,min):
        """
        USES LAPLACE APPROXIMATION TO CALCULATE THE BAYESIAN MODEL POSTERIOR
        """
        Sigma = self.getLaplaceCovar(min)
        n_params = self.vd.getNumberScales()
        ModCompl = 0.5*n_params*SP.log(2*SP.pi)+0.5*SP.log(SP.linalg.det(Sigma))
        RV = min['LML']+ModCompl
        return RV


    def getEmpTraitCovar(self):
        """
        Returns the empirical trait covariance matrix
        """
        if self.P==1:
            out=self.Y[self.Iok].var()
        else:
            out=SP.cov(self.Y[self.Iok].T)
        return out


    def getEmpTraitCorrCoef(self):
        """
        Returns the empirical trait correlation matrix
        """
        cov = self.getEmpTraitCovar()
        stds=SP.sqrt(cov.diagonal())[:,SP.newaxis]
        RV = cov/stds/stds.T
        return RV
    
    
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
    
            if verbose: print p
    
        sth = {}
        sth['varg']  = varg
        sth['varn']  = varn
        sth['fixed'] = fixed

        return sth


    def empCov2params(self):
        """
        calculate parameters from the empirical covariance matrix
        """
        
        assert self.noisPos!=None, 'No noise element'
        
        noise_type = self.vd.getTerm(self.noisPos).getTraitCovar().getName()
        
        assert noise_type in ['CFreeFormCF','SumCF'], 'not supported for %s noise matrices' % noise_type
        
        empCovar = self.getEmpTraitCovar()
        
        if self.cache['Lparams']==None:
            if noise_type=='CFreeFormCF':
                L = SP.linalg.cholesky(empCovar).T
                self.cache['Lparams'] = SP.concatenate([L[i,:(i+1)] for i in range(self.P)])
            else:
                if (self.P==2):
                    chia=empCovar[0,0]
                    chib=empCovar[1,1]
                    corr=empCovar[0,1]/SP.sqrt(chia*chib)
                    c2 = 0.5*(chia+chib-SP.sqrt((chia+chib)**2-4*chia*chib*(1-corr**2)))
                    a2 = chia - c2
                    b2 = chib - c2
                    self.cache['Lparams'] = SP.sqrt(abs(SP.array([a2,b2,c2])))
                    if corr<0:   self.cache['Lparams'][1] = -self.cache['Lparams'][1]
                else:
                    S, U = SP.linalg.eigh(empCovar)
                    k = SP.argmax(S)
                    a = SP.sqrt(S[k])*U[:,k]
                    c = (empCovar-SP.dot(a[:,SP.newaxis],a[:,SP.newaxis].T)).diagonal()
                    c = SP.sqrt(c)
                    self.cache['Lparams'] = SP.concatenate((a,c))
    
        return self.cache['Lparams']


    def herit2params(self):
        """
        calculates parameters from the single trait model
        """
        
        assert self.noisPos!=None, 'No noise element'
        assert self.n_terms==2,    'number of element must be 2'
        assert self.P>=2,          'number of phenotypes must be >=2'
        
        noise_type = self.vd.getTerm(self.noisPos).getTraitCovar().getName()
        assert noise_type in ['CFreeFormCF','SumCF'], 'not supported for %s noise matrices' % noise_type
        
        sth = self.estimateHeritabilities(self.vd.getTerm(abs(self.noisPos-1)).getK())
    
        if self.cache['paramsST']==None:
            self.cache['paramsST'] = SP.zeros(0)
            for i in range(self.n_terms):
                if i==self.noisPos: var = sth['varn']
                else:               var = sth['varg']
                if noise_type=='CFreeFormCF':
                    L = SP.diag(SP.sqrt(var))
                    temp = SP.concatenate([L[i,:(i+1)] for i in range(self.P)])
                else:
                    if self.P==2:
                        temp = SP.array([var[0]-var.min(),var[0]-var.min(),var.max()-var.min()])
                        temp = SP.sqrt(temp)
                    else:
                        temp = SP.concatenate((SP.zeros(self.P),SP.sqrt(var)))
                self.cache['paramsST'] = SP.concatenate((self.cache['paramsST'],temp))
            
        return self.cache['paramsST']

    
    def initEmpirical(self):
        """
        init using parameters from the empirical covariance
        """
        
        assert self.noisPos!=None, 'No noise element'
        
        for term_i in range(self.n_terms):
            if term_i==self.noisPos:
                if self.P==1:
                    self.setScales(SP.array([SP.sqrt(self.getEmpTraitCovar())]),term_i)
                else:
                    params  = self.empCov2params()
                    self.setScales(params,term_i)
            else:
                n_scales = self.vd.getTerm(term_i).getNumberScales()
                params = SP.zeros(n_scales)
                self.setScales(params,term_i)
    
    
    def initSingleTrait(self):
        """
        init using parameters from the single trait model
        """
        
        assert self.noisPos!=None, 'No noise element'
        assert self.n_terms==2, 'number of element must be 2'
        assert self.P>=2, 'number of phenotypes must be >=2'

        params  = self.herit2params()

        self.setScales(params)

            

    def perturbScales(self,std=1e-3):
        """
        perturb the current values of the scales by N(0,std**2)
        """
        
        params = self.getScales()
        params+= std*SP.randn(params.shape[0])
        self.setScales(params)

    """
    CODE FOR PREDICTIONS
    """
    
    def setKstar(self,term_i,Ks):
    
        assert Ks.shape[0]==self.N
    
        #if Kss!=None:
            #assert Kss.shape[0]==Ks.shape[1]
            #assert Kss.shape[1]==Ks.shape[1]

        self.vd.getTerm(term_i).getKcf().setK0cross(Ks)
    
    
    
    def predictMean(self,term_i=None):
        
        assert self.noisPos!=None,      'No noise element'
        assert self.init,               'GP not initialised'
        assert term_i!=self.noisPos,    'Noise term predictions have mean 0'
        
        KiY = self.gp.agetKEffInvYCache()
        
        if self.fast==False:
            KiY = KiY.reshape(self.P,self.N).T
        
        if term_i!=None:
        
            Rs = self.vd.getTerm(term_i).getKcf().Kcross(SP.zeros(1))
            Ymean = SP.dot(Rs.T,KiY)
            if self.P>1:
                C = self.vd.getTerm(term_i).getTraitCovar().K()
                temp = SP.dot(Ymean,C)
        else:
            Ymean = None
            for term_i in range(self.n_terms):
                if term_i!=self.noisPos:
                    if Ymean==None:      Ymean  = self.predictMean(term_i)
                    else:               Ymean += self.predictMean(term_i)
            Ymean+=self.getFixed()[:,0]

        return Ymean
        

        








