import scipy as SP
import numpy as NP
import scipy.linalg as LA
import scipy.optimize as OPT
import scipy.stats as st
import pdb


# log of 2pi
L2pi = 1.8378770664093453
def nLLeval(ldelta,UY,UX,S,MLparams=False):
    """evaluate the negative LL of a LMM with kernel USU.T"""
    delta=SP.exp(ldelta);
    n,d=UX.shape;
    Sdi=S+delta;
    ldet=SP.sum(NP.log(Sdi));
    Sdi=1.0/Sdi;
    XSdi=UX.T*SP.tile(Sdi,(d,1));
    XSX=SP.dot(XSdi,UX);
    XSY=SP.dot(XSdi,UY);
    beta=LA.lstsq(XSX,XSY);
    res=UY-SP.dot(UX,beta[0]);
    res*=res;
    res*=Sdi;
    sigg2=SP.sum(res)/n;
    nLL=0.5*(n*L2pi+ldet+n+n*NP.log(sigg2));
    if MLparams:
        return nLL, beta[0], sigg2;
    else:
        return nLL;

def optdelta(UY,UX,S,ldeltanull=None,numintervals=100,ldeltamin=-10.0,ldeltamax=10.0):
    """find the optimal delta"""
    if ldeltanull==None:
        nllgrid=SP.ones(numintervals+1)*SP.inf;
        ldeltagrid=SP.arange(numintervals+1)/(numintervals*1.0)*(ldeltamax-ldeltamin)+ldeltamin;
        nllmin=SP.inf;
        for i in SP.arange(numintervals+1):
            nllgrid[i]=nLLeval(ldeltagrid[i],UY,UX,S);
            if nllgrid[i]<nllmin:
                nllmin=nllgrid[i];
                ldeltaopt_glob=ldeltagrid[i];
        foundMin=False
        for i in SP.arange(numintervals-1)+1:
            continue
            ee = 1E-8
            #carry out brent optimization within the interval
            if ((nllgrid[i-1]-nllgrid[i])>ee) and ((nllgrid[i+1]-nllgrid[i])>1E-8):
                foundMin = True
                ldeltaopt,nllopt,iter,funcalls = OPT.brent(nLLeval,(UY,UX,S),(ldeltagrid[i-1],ldeltagrid[i],ldeltagrid[i+1]),full_output=True);
                if nllopt<nllmin:
                    nllmin=nllopt;
                    ldeltaopt_glob=ldeltaopt;
    else:
        ldeltaopt_glob=ldeltanull;
    return ldeltaopt_glob;

def estimateBeta(X,Y,K,C=None,addBiasTerm=False,numintervals0=100,ldeltamin0=-5.0,ldeltamax0=5.0):
    """ compute all pvalues
    If numintervalsAlt==0 use EMMA-X trick (keep delta fixed over alternative models)
    """
    n,s=X.shape;
    n_pheno=Y.shape[1];
    S,U=LA.eigh(K);
    UY=SP.dot(U.T,Y);
    UX=SP.dot(U.T,X);
    if (C==None):
        Ucovariate=SP.dot(U.T,SP.ones([n,1]));
    else:
        if (addBiasTerm):
            C_=SP.concatenate((C,SP.ones([n,1])),axis=1)
            Ucovariate=SP.dot(U.T,C_);
        else:
            Ucovariate=SP.dot(U.T,C);
    n_covar=Ucovariate.shape[1];
    beta = SP.empty((n_pheno,s,n_covar+1));
    LL=SP.ones((n_pheno,s))*(-SP.inf);
    ldelta=SP.empty((n_pheno,s));
    sigg2=SP.empty((n_pheno,s));
    pval=SP.ones((n_pheno,s))*(-SP.inf);
    for phen in SP.arange(n_pheno):
        UY_=UY[:,phen];
        ldelta[phen]=optdelta(UY_,Ucovariate,S,ldeltanull=None,numintervals=numintervals0,ldeltamin=ldeltamin0,ldeltamax=ldeltamax0);
        for snp in SP.arange(s):
            UX_=SP.hstack((UX[:,snp:snp+1],Ucovariate));
            nLL_, beta_, sigg2_=nLLeval(ldelta[phen,snp],UY_,UX_,S,MLparams=True);
            beta[phen,snp,:]=beta_;
            sigg2[phen,snp]=sigg2_;
            LL[phen,snp]=-nLL_;
    return beta, ldelta

def train_associations(X,Y,K,C=None,addBiasTerm=False,numintervalsAlt=0,ldeltaminAlt=-1.0,ldeltamaxAlt=1.0,numintervals0=100,ldeltamin0=-5.0,ldeltamax0=5.0, calc_pval=True):
    """ compute all pvalues
    If numintervalsAlt==0 use EMMA-X trick (keep delta fixed over alternative models)
    """
    n,s=X.shape;
    n_pheno=Y.shape[1];
    S,U=LA.eigh(K);
    UY=SP.dot(U.T,Y);
    UX=SP.dot(U.T,X);
    if (C==None):
        Ucovariate=SP.dot(U.T,SP.ones([n,1]));
    else:
        if (addBiasTerm):
            C_=SP.concatenate((C,SP.ones([n,1])),axis=1)
            Ucovariate=SP.dot(U.T,C_);
        else:
            Ucovariate=SP.dot(U.T,C);
    n_covar=Ucovariate.shape[1];
    beta = SP.empty((n_pheno,s,n_covar+1));
    beta0 = SP.empty((n_pheno,n_covar));
    LL=SP.ones((n_pheno,s))*(-SP.inf);
    LL0=SP.ones((n_pheno))*(-SP.inf);
    ldelta=SP.empty((n_pheno,s));
    ldelta0=SP.empty(n_pheno);
    sigg2=SP.empty((n_pheno,s));
    sigg20=SP.empty((n_pheno));
    pval=SP.ones((n_pheno,s))*(-SP.inf);
    for phen in SP.arange(n_pheno):
        UY_=UY[:,phen];
        ldelta0[phen]=optdelta(UY_,Ucovariate,S,ldeltanull=None,numintervals=numintervals0,ldeltamin=ldeltamin0,ldeltamax=ldeltamax0);
        print(('log(delta) was fitted to', ldelta0))
        #print ldelta0
        #print "ldelta0 \n" 
        nLL0_, beta0_, sigg20_=nLLeval(ldelta0[phen],UY_,Ucovariate,S,MLparams=True);
        beta0[phen,:]=beta0_;
        sigg20[phen]=sigg20_;
        LL0[phen]=-nLL0_;
        for snp in SP.arange(s):
            UX_=SP.hstack((UX[:,snp:snp+1],Ucovariate));
            if numintervalsAlt==0: #EMMA-X trick #fast version, no refitting of detla
                ldelta[phen,snp]=ldelta0[phen];                
            else: #fit delta
                ldelta[phen,snp]=optdelta(UY_,UX_,S,ldeltanull=None,numintervals=numintervalsAlt,ldeltamin=ldelta0[phen]+ldeltaminAlt,ldeltamax=ldelta0[phen]+ldeltamaxAlt);
            nLL_, beta_, sigg2_=nLLeval(ldelta[phen,snp],UY_,UX_,S,MLparams=True);
            beta[phen,snp,:]=beta_;
            sigg2[phen,snp]=sigg2_;
            LL[phen,snp]=-nLL_;
    #reshaping of LL0
    LL0 = LL0[:,SP.newaxis]
    lods = LL-LL0
    if calc_pval:
        arg2 = st.chi2.sf(2*(lods),1)
    else:
        arg2 = ldelta 
    #return LL0, LL, pval, ldelta0, sigg20, beta0, ldelta, sigg2, beta
    return lods, arg2

def train_interact(X,Y,K,interactants=None,covariates=None,addBiasTerm=True,numintervalsAlt=0,ldeltaminAlt=-1.0,ldeltamaxAlt=1.0,numintervals0=10,ldeltamin0=-5.0,ldeltamax0=5.0):
    """ compute all pvalues
    If numintervalsAlt==0 use EMMA-X trick (keep delta fixed over alternative models)
    difference to previous model: Ux and Ucovariate are recomputed for every SNP
    """
    n,s=X.shape;
    n_pheno=Y.shape[1];
    S,U=LA.eigh(K);
    UY=SP.dot(U.T,Y);
    UX=SP.dot(U.T,X);
    if (covariates==None):
        covariates = SP.ones([n,0])
    if (addBiasTerm):
        covariates=SP.concatenate((covariates,SP.ones([n,1])),axis=1)
    #Ucovariates
    Ucovariate=SP.dot(U.T,covariates);

    #Uinteractants
    Uinteractants = SP.dot(U.T,interactants)
    n_covar=covariates.shape[1]
    n_inter=interactants.shape[1]
    #weights
    #foreground: covaraits + SNP + interactions 
    beta = SP.empty((n_pheno,s,1+n_covar+2*n_inter));
    #background: covariates + direct SNP effect
    beta0 = SP.empty((n_pheno,s,1+n_covar+n_inter));
    LL=SP.ones((n_pheno,s))*(-SP.inf);
    LL0=SP.ones((n_pheno,s))*(-SP.inf);
    ldelta=SP.empty([n_pheno,s]);
    ldelta0=SP.empty([n_pheno,s]);
    sigg2=SP.empty((n_pheno,s));
    sigg20=SP.empty((n_pheno,s));
    pval=SP.ones((n_pheno,s))*(-SP.inf);
    for snp in SP.arange(s):
        #loop through all SNPs
        #1. snp-specific backgroud model SNP effect + covaraites + interactants
        Ucovariates_=SP.hstack((UX[:,snp:snp+1],Uinteractants,Ucovariate))
        #2. snp-specific foreground model
        #interactions
        Xi_ = X[:,snp:snp+1]*interactants
        #transform
        UXi_ = SP.dot(U.T,Xi_)
        #stack: interactions, interactants (main) SNPs (main) covariates (if any)
        UX_  = SP.hstack((UXi_,Ucovariates_))
        for phen in SP.arange(n_pheno):
            print(phen)
            #loop through phenoptypes
            #get transformed Y
            UY_=UY[:,phen]        
            #1. fit background model
            ldelta0[phen,snp]=optdelta(UY_,Ucovariates_,S,ldeltanull=None,numintervals=numintervals0,ldeltamin=ldeltamin0,ldeltamax=ldeltamax0);
            nLL0_, beta0_, sigg20_=nLLeval(ldelta0[phen,snp],UY_,Ucovariates_,S,MLparams=True);
            beta0[phen,snp,:]=beta0_;
            sigg20[phen,snp]=sigg20_;
            LL0[phen,snp]=-nLL0_;            

            #2. fit foreground model
            if numintervalsAlt==0: #EMMA-X trick #fast version, no refitting of detla
                ldelta[phen,snp]=ldelta0[phen,snp]                
            else: #fit delta
                ldelta[phen,snp]=optdelta(UY_,UX_,S,ldeltanull=None,numintervals=numintervalsAlt,ldeltamin=ldelta0[phen,snp]+ldeltaminAlt,ldeltamax=ldelta0[phen,snp]+ldeltamaxAlt);
            nLL_, beta_, sigg2_=nLLeval(ldelta[phen,snp],UY_,UX_,S,MLparams=True);
            beta[phen,snp,:]=beta_;
            sigg2[phen,snp]=sigg2_;
            LL[phen,snp]=-nLL_;
    pval = st.chi2.sf(2*(LL-LL0),1)
    return LL0, LL, pval, ldelta0, sigg20, beta0, ldelta, sigg2, beta



def train_interactX(X,Y,K,interactants=None,covariates=None,addBiasTerm=True,numintervalsAlt=0,ldeltaminAlt=-1.0,ldeltamaxAlt=1.0,numintervals0=10,ldeltamin0=-5.0,ldeltamax0=5.0):
    """ compute all pvalues
    If numintervalsAlt==0 use EMMA-X trick (keep delta fixed over alternative models)
    difference to previous model: Ux and Ucovariate are recomputed for every SNP
    """
    n,s=X.shape;
    n_pheno=Y.shape[1];
    S,U=LA.eigh(K);
    UY=SP.dot(U.T,Y);
    UX=SP.dot(U.T,X);
    if (covariates==None):
        covariates = SP.ones([n,0])
    if (addBiasTerm):
        covariates=SP.concatenate((covariates,SP.ones([n,1])),axis=1)
    #Ucovariates
    Ucovariate=SP.dot(U.T,covariates);

    #Uinteractants
    Uinteractants = SP.dot(U.T,interactants)
    n_covar=covariates.shape[1]
    n_inter=interactants.shape[1]
    #weights
    #foreground: covaraits + SNP + interactions 
    beta = SP.empty((n_pheno,s,1+n_covar+2*n_inter));
    #background: covariates + direct SNP effect
    beta0 = SP.empty((n_pheno,s,1+n_covar+n_inter));
    LL=SP.ones((n_pheno,s))*(-SP.inf);
    LL0=SP.ones((n_pheno,s))*(-SP.inf);
    ldelta=SP.empty([n_pheno,s]);
    ldelta0=SP.empty([n_pheno,s]);
    sigg2=SP.empty((n_pheno,s));
    sigg20=SP.empty((n_pheno,s));
    pval=SP.ones((n_pheno,s))*(-SP.inf);
    #0. fit 0 model on phenotypes and covariates alone
    for phen in SP.arange(n_pheno):
        #fit if phen is visited the first time
        #loop through phenoptypes
        #get transformed Y
        UY_=UY[:,phen]        
        #1. fit background model to set delta
        ldelta0[phen,:]=optdelta(UY_,Ucovariate,S,ldeltanull=None,numintervals=numintervals0,ldeltamin=ldeltamin0,ldeltamax=ldeltamax0);
            
    #1. loop through all snps
    for snp in SP.arange(s):
        #loop through all SNPs
        #1. snp-specific backgroud model SNP effect + covaraites + interactants
        Ucovariates_=SP.hstack((UX[:,snp:snp+1],Uinteractants,Ucovariate))
        #2. snp-specific foreground model
        #interactions
        Xi_ = X[:,snp:snp+1]*interactants
        #transform
        UXi_ = SP.dot(U.T,Xi_)
        #stack: interactions, interactants (main) SNPs (main) covariates (if any)
        UX_  = SP.hstack((UXi_,Ucovariates_))

        for phen in SP.arange(n_pheno):
            UY_=UY[:,phen]        
            #loop through all phenotypes
            #emmaX trick
            ldelta[phen,snp]=ldelta0[phen,snp]
            #evluate background and foreground
            #null model
            nLL0_, beta0_, sigg20_=nLLeval(ldelta0[phen,snp],UY_,Ucovariates_,S,MLparams=True)
            beta0[phen,snp,:]=beta0_
            sigg20[phen,snp]=sigg20_
            LL0[phen,snp]=-nLL0_                   
            #foreground model
            nLL_, beta_, sigg2_=nLLeval(ldelta[phen,snp],UY_,UX_,S,MLparams=True)
            beta[phen,snp,:]=beta_
            sigg2[phen,snp]=sigg2_
            LL[phen,snp]=-nLL_
    pval = st.chi2.sf(2*(LL-LL0),1)
    return LL0, LL, pval, ldelta0, sigg20, beta0, ldelta, sigg2, beta


def run_interact(Y, intA, intB, covs, K):
    """ Calculate pvalues for the nested model of including a multiplicative term between intA and intB into the additive model """
    [N, Ny] = Y.shape

    Na = intA.shape[1] # number of interaction terms 1
    Nb = intB.shape[1] # number of interaction terms 2
    
    S,U=LA.eigh(K);
    UY=SP.dot(U.T,Y);
    UintA=SP.dot(U.T,intA);
    UintB=SP.dot(U.T,intB);
    Ucovs=SP.dot(U.T,covs);
    # for each snp/gene/factor combination, run a lod
    # snps need to be diced bc of missing values - iterate over them, else in arrays
    lods = SP.zeros([Na, Nb, Ny])

    #add mean column:
    if covs is None: covs = SP.ones([Ny,1])

    # for each pair of interacting terms
    for a in range(Na):
        for b in range(Nb):
            # calculate additive and interaction terms
            C = SP.concatenate((Ucovs, UintA[:,a:a+1], UintB[:,b:b+1]))
            X = intA[:,a:a+1]*intB[:,b:b+1]
            UX = SP.dot(U.T,X);
            UX = SP.concatenate((UX, C))
            for phen in SP.arange(Ny):
                UY_=UY[:,phen];
                nllnull,ldeltanull=optdelta(UY_,C,S,ldeltanull=None,numintervals=10,ldeltamin=-5.0,ldeltamax=5.0);
                nllalt,ldeltaalt=optdelta(UY_,UX,S,ldeltanull=ldeltanull,numintervals=100,ldeltamin=-5.0,ldeltamax=5.0);
                lods[a,b,phen] = nllalt-nllalt;
    return lods
