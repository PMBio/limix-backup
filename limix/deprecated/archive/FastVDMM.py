import sys
sys.path.append('./..')
sys.path.append('./../../..')

import scipy as SP
import scipy.linalg
import limix
import pdb
import time

#TODO: what is this script? Can we remove for the release?
#development stuff should go into separate branches


def kroneckerApprox(S,K,Gamma):
    
    import scipy.optimize
    
    def fitKronApprox(a):
        Sbg = SP.zeros_like(S[0])
        Kbg = SP.zeros_like(K[0])
        for i in range(len(S)):  Sbg+= a[i]*S[i]
        for i in range(len(K)):  Kbg+= a[i+len(S)]*K[i]
        Gamma1 = SP.kron(Sbg,Kbg)
        return ((Gamma-Gamma1)**2).sum()

    a0 = SP.concatenate((1/float(len(S))*SP.ones(len(S)),1/float(len(K))*SP.ones(len(K))))
    res = SP.optimize.fmin_l_bfgs_b(fitKronApprox,a0,approx_grad = 1, factr=10., pgtol=1e-08, maxfun=float('Inf'))
    a = res[0]
    Sbg = SP.zeros_like(S[0])
    Kbg = SP.zeros_like(K[0])
    for i in range(len(S)):  Sbg+= a[i]*S[i]
    for i in range(len(K)):  Kbg+= a[i+len(S)]*K[i]

    return Sbg, Kbg


class CFastVDMM:
    """
    helper function for variance decomposition in limix
    This class mainly takes care of initialization and interpreation of results
    """
    
    """
    Methods:
        __init__(self,Y,T,F,C,F):    Constructor
        initialise:                  Initialise parameters
        fit:                         Fit phenos and returns the minimum with all the info
        fit_ntimes:                  Fit phenos ntimes with different random initialization and returns the minima order with respect to gp.LML
        getGP:                       Return the GP of the limix variacne decomposition class
        getParams:                   Returns the stack vector of the parameters of the trait covariances C_i
        getEmpTraitCov:              Returns the empirical trait covariance
        getEstTraitCov:              Returns the total trait covariance in the GP by summing up the trait covariances C_i
        getCovParams:                Calculates the inverse hessian of the -loglikelihood with respect to the parameters: this constitutes the covariance matrix of the posterior over paramters under laplace approximation
        estimateHeritabilities:      It fits the model with 1 global fixed effects and covariance matrix h1[p]*K+h2[p] for each trait p and returns the vectors h1 and h2
        exportMin:                   Export information about the min in the given h5py file or group
        exportMins:                  Export information about all the min in the given h5py file or group

    """
    
    def __init__(self,Y,C1,C2,K1,K2=None):
        """
        Args:
            Y:  NxP phenotype matrix
            C:  list of trait covariances
            K1: kernel matrix 1
            K2: kernel matrix 2
        """
        
        self.N=Y.shape[0]
        self.P=Y.shape[1]
        
        #Diagonalization of Part II
        if K2 == None:
            K2   = SP.eye(self.N)
            U2   = SP.eye(self.N)
            S2is = SP.eye(self.N)
            logdet_compl = 0
            K = K1
        else:
            eigen2, U2 = SP.linalg.eigh(K2)
            eigen2= SP.array(eigen2,dtype=float)
            S2is = SP.diag(1/SP.sqrt(eigen2))
            logdet_compl = self.P*SP.log(eigen2).sum()
            K = SP.dot(S2is,SP.dot(U2.T,SP.dot(K1,SP.dot(U2,S2is))))
                
        #Diagonalization of Part I
        eigen, U = SP.linalg.eigh(K)
        eigen = SP.array(eigen,dtype=float)
        U= SP.array(U,dtype=float)
        S   = SP.diag(eigen)
                
        # pheno transformation
        Yt = SP.zeros_like(Y.T)
        Upheno = SP.dot(U.T,SP.dot(S2is,U2.T))
        for p in range(self.P):
            Yt[p,:] = SP.dot(Upheno,Y[:,p])
        
        self.C1 = self.buildCF(C1)
        self.C2 = self.buildCF(C2)
        self.Yt = Yt
        self.eigen = eigen
        pass
    
    def buildCF(self,C):
        # Initialize CTraitCFs and wrap them in a Sum of matrices
        trait = SP.array(list(range(self.P)))[:,SP.newaxis]
        if type(C)!=list:
            Cout=C
            if Cout.getName()[0:2]=='CT':    Cout.setX(trait)
        else:
            Cout = limix.CSumCF()
            n_terms = len(C)
            for term_i in range(n_terms):
                if C[term_i].getName()[0:2]=='CT':    C[term_i].setX(trait)
                Cout.addCovariance(C[term_i])
        return Cout

    
    
    def initialise(self):
        """
        get random initialization of variances based on the empirical trait variance
        
        EmpVarY=self.getEmpTraitVar()
        temp=SP.rand(self.n_terms,self.P)
        N=temp.sum(0)
        temp=temp/N*EmpVarY
        for term_i in range(self.n_terms):
            if self.C[term_i].getName()=='CTDenseCF':
                params=(2*SP.rand(self.P)-1)*SP.sqrt(temp[term_i,:])
                self.C[term_i].setParams(params)
            elif self.C[term_i].getName()=='CTFixedCF':
                params=SP.array([SP.sqrt(temp[term_i,:].mean())])
                self.C[term_i].setParams(params)
            elif self.C[term_i].getName()=='CTDiagonalCF':
                params=SP.sqrt(temp[term_i,:])
                self.C[term_i].setParams(params)
            else:
                print 'Not implemented for %s' % self.C[term_i].getName()
                break
        """

    
    def fit(self,Params0=None,grad_threshold=1e-2):
        """
        fit a variance component model with the predefined design and the initialization and returns all the results
        """

        # GPVD initialization
        lik = limix.CLikNormalNULL()
        # Initial Params
        if Params0==None:
            n_params = self.C1.getNumberParams()
            n_params+= self.C2.getNumberParams()
            Params0 = SP.rand(n_params)
        # MultiGP Framework
        covar = []
        gp    = []
        mean  = []
        for i in range(self.N):
            covar.append(limix.CLinCombCF())
            covar[i].addCovariance(self.C1)
            covar[i].addCovariance(self.C2)
            coeff = SP.array([self.eigen[i],1])
            covar[i].setCoeff(coeff)
            mean.append(limix.CLinearMean(self.Yt[:,i],SP.eye(self.P)))
        gpVD = limix.CGPvarDecomp(covar[0],lik,mean[0],SP.ones(self.N),self.P,self.Yt,Params0)
        for i in range(self.N):
            gp.append(limix.CGPbase(covar[i],lik,mean[i]))
            gp[i].setY(self.Yt[:,i])
            gpVD.addGP(gp[i])
                
        # Optimization
        gpVD.initGPs()
        gpopt = limix.CGPopt(gpVD)
        LML0=-1.0*gpVD.LML()
        start_time = time.time()
        conv = gpopt.opt()
        time_train = time.time() - start_time
        LML=-1.0*gpVD.LML()
        LMLgrad = SP.linalg.norm(gpVD.LMLgrad()['covar'])
        Params = gpVD.getParams()['covar']
    
        # Check whether limix::CVarianceDecomposition.train() has converged
        if conv!=True or LMLgrad>grad_threshold or Params.max()>10:
            print('limix::CVarianceDecomposition::train has not converged')
            res=None
        else:
            res = {
                'Params0':          Params0,
                'Params':           Params,
                'LML':              SP.array([LML]),
                'LML0':             SP.array([LML0]),
                'LMLgrad':          SP.array([LMLgrad]),
                'time_train':       SP.array([time_train]),
                }
        return res
        pass
            

    
    
    def fit_ntimes(self,ntimes=10,grad_threshold=1e-2,dist_mins=1e-2):
        """
        fit phenos ntimes with different random initialization and returns the minima order with respect to gp.LML
        

        optima=[]
        LML=SP.zeros((1,0))
        
        for i in range(ntimes):
            
            print ".. Minimization %d" % i
            
            self.initialise()
            min=self.fit(grad_threshold=grad_threshold)
        
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
    
         # Order the list optima with respect to LML the first optimum has highest LML
        optima1=[]
        index = LML.argsort(axis = 1)[0,:][::-1]
        for i in range(len(optima)):
            optima1.append(optima[index[i]])
    
        return optima1
        """

    
    def getGP(self):
        """
        Returns the GP of the limix class CVarianceDecomposition
        
        vt = limix.CVarianceDecomposition()
        vt.setPheno(self.Y)
        vt.setFixed(self.F)
        for term_i in range(self.n_terms):
            vt.addCVTerm(self.C[term_i],self.K[term_i],self.T[term_i])
        vt.initGP()
        gp=vt.getGP()
        return gp
        """

    def getParams(self):
        """
        Returns the Parameters
        
        params=SP.concatenate([self.C[term_i].agetScales() for term_i in range(self.n_terms)])
        return params
        """

    def getEstTraitCov(self):
        """
        Returns the estimated trait covariance matrix
        
        TraitCovar=SP.zeros((self.P,self.P))
        for term_i in range(self.n_terms):
            TraitCovar+=self.C[term_i].getK0()
        return TraitCovar
        """

    def getCovParams(self,min):
        """
        USES LAPLACE APPROXIMATION TO CALCULATE THE COVARIANCE MATRIX OF THE OPTIMIZED PARAMETERS
        
        gp=min['gp']
        ParamMask=gp.getParamMask()['covar']
        std=SP.zeros(ParamMask.sum())
        H=gp.LMLhess(["covar"])
        It= (ParamMask[:,0]==1)
        H=H[It,:][:,It]
        Sigma = SP.linalg.inv(H)
        return Sigma
        """
    
    def getModelPosterior(self,min,Sigma=None):
        """
        USES LAPLACE APPROXIMATION TO CALCULATE THE BAYESIAN MODEL POSTERIOR
        
        if Sigma==None:
            Sigma = self.getCovParams(min)
        
        n_params = 0
        for term_i in range(self.n_terms):
            n_params += self.C[term_i].getNumberParams()
        
        ModCompl = 0.5*n_params*SP.log(2*SP.pi)+0.5*SP.log(SP.linalg.det(Sigma))
        
        RV = min['LML']+ModCompl
            
        return RV
        """
    

    def getEmpTraitCov(self):
        """
        Returns the empirical trait covariance matrix
        
        Y1=(self.Y).reshape((self.P,self.N))
        RV=SP.cov(Y1)
        return RV
        """
    
    def getEmpTraitVar(self):
        """
        Returns the vector of empirical trait variances
        
        if self.P==1:
            RV = self.getEmpTraitCov()
        else:
            RV = self.getEmpTraitCov().diagonal()
        return RV
        """




    def estimateHeritabilities(self,Kpop):
        """
        It fits the model with 1 fixed effects and covariance matrix h1[p]*K+h2[p] for each trait p and return the vectors h1 and h2
        
    
        h1 = SP.zeros(self.P)
        h2 = SP.zeros(self.P)
    
        for p in range(self.P):
            It = (self.T[0][:,0]==p)
            K= Kpop[It,:][:,It]
            N= K.shape[0]
            y=self.Y[It]
            h1[p], h2[p]=limix.CVarianceDecomposition.aestimateHeritability(y,SP.ones(N),K)
            if h1[p]<1e-6:   h1[p]=1e-6
            if h2[p]<1e-6:   h2[p]=1e-6
    
        return h1, h2
        """
    
    def exportMin(self,min,f,counter=0,laplace=0):
        """
        Export the min in the given h5py file or group    
        
        f.create_dataset("Params0",data=min["Params0"])
        f.create_dataset("LML0",data=min["LML0"])
        f.create_dataset("Params",data=min["Params"])
        f.create_dataset("LML",data=min["LML"])
        f.create_dataset("LMLgrad",data=min["LMLgrad"])
        f.create_dataset("time_train",data=min["time_train"])
        if counter!=0:
            f.create_dataset("counter",data=min["counter"])
        if laplace!=0:
            covParams = self.getCovParams(min)
            f.create_dataset("covParams",data=covParams)
            f.create_dataset("modelPosterior",data=self.getModelPosterior(min,Sigma=covParams))
        """
        
    
    def exportMins(self,mins,f,laplace=0):
        """
        Export all the min in the given h5py file or group  
        
        for min_i in range(len(mins)):
            g=f.create_group('min%d'%min_i)
            self.exportMin(mins[min_i],g,counter=1,laplace=laplace)
        """




