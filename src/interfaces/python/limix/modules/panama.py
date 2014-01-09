"""
PANAMA mododule in limix
"""
import limix.modules.qtl as qtl
import limix.utils.fdr as fdr
import limix
import scipy as SP
import pdb
import pylab as PL
import scipy.linalg as linalg
import time

def PCA(Y, components):
    """run PCA, retrieving the first (components) principle components
    return [s0, eig, w0]
    s0: factors
    w0: weights
    """
    sv = linalg.svd(Y, full_matrices=0);
    [s0, w0] = [sv[0][:, 0:components], SP.dot(SP.diag(sv[1]), sv[2]).T[:, 0:components]]
    v = s0.std(axis=0)
    s0 /= v;
    w0 *= v;
    return [s0, w0]


class PANAMA:
    """PANAMA class"""
    def __init__(self,data=None,X=None,Y=None,Kpop=None,use_Kpop=True,standardize=True):
        """
        data: data object to feed form
        X: alternatively SNP data
        Y: alternativel expression data 
        Kpop: Kpop
        use_Kpop: use Kpop kernel? (True)
        standardize (True)
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
        else:
            self.Kpop = SP.dot(self.X,self.X.T)
            self.Kpop /= self.Kpop.diagonal().mean()
        if standardize:
            self.Y -= self.Y.mean(axis=0)
            self.Y /= self.Y.std(axis=0)
        self.N = self.Y.shape[0]
        self.P = self.Y.shape[1]
        if self.X is not None:
            self.S = self.X.shape[1]
            assert self.X.shape[0]==self.Y.shape[0], 'data size missmatch'
        pass

    def train(self,K=20,Kpop=True):
        """train panama module"""
        if 0:
            covar  = limix.CCovLinearISO(K)
            ll  = limix.CLikNormalIso()
            X0 = SP.random.randn(self.N,K)
            X0 = PCA(self.Y,K)[0]
            X0 /= SP.sqrt(K)
            covar_params = SP.array([1.0])
            lik_params = SP.array([1.0])

            hyperparams = limix.CGPHyperParams()
            hyperparams['covar'] = covar_params
            hyperparams['lik'] = lik_params
            hyperparams['X']   = X0
        
            constrainU = limix.CGPHyperParams()
            constrainL = limix.CGPHyperParams()
            constrainU['covar'] = +5*SP.ones_like(covar_params);
            constrainL['covar'] = 0*SP.ones_like(covar_params);
            constrainU['lik'] = +5*SP.ones_like(lik_params);
            constrainL['lik'] = 0*SP.ones_like(lik_params);

        if 1:
            covar  = limix.CSumCF()
            covar_1 =  limix.CCovLinearISO(K)
            covar.addCovariance(covar_1)
            covar_params = [1.0]

            if self.use_Kpop:
                covar_2 =  limix.CFixedCF(self.Kpop)
                covar.addCovariance(covar_2)
                covar_params.append(1.0)

            ll  = limix.CLikNormalIso()
            X0 = SP.random.randn(self.N,K)
            X0 = PCA(self.Y,K)[0]
            X0 /= SP.sqrt(K)
            covar_params = SP.array(covar_params)
            lik_params = SP.array([1.0])

            hyperparams = limix.CGPHyperParams()
            hyperparams['covar'] = covar_params
            hyperparams['lik'] = lik_params
            hyperparams['X']   = X0
        
            constrainU = limix.CGPHyperParams()
            constrainL = limix.CGPHyperParams()
            constrainU['covar'] = +5*SP.ones_like(covar_params);
            constrainL['covar'] = 0*SP.ones_like(covar_params);
            constrainU['lik'] = +5*SP.ones_like(lik_params);

            
        gp=limix.CGPbase(covar,ll)
        gp.setY(self.Y)
        gp.setX(X0)
        lml0 = gp.LML(hyperparams)
        dlml0 = gp.LMLgrad(hyperparams)        
        gpopt = limix.CGPopt(gp)
        gpopt.setOptBoundLower(constrainL);
        gpopt.setOptBoundUpper(constrainU);

        t1 = time.time()
        gpopt.opt()
        t2 = time.time()
        self.Kconfounder = covar.K()
        #normalize
        self.Kconfounder/= self.Kconfounder.trace()/self.Kconfounder.shape[0]
        self.Xconfounder = gp.getParams()['X']
        #store relative variance of Kpop XX
        V = SP.zeros([3])
        V[0] = covar_1.K().diagonal().mean()
        if self.use_Kpop:
            V[1] = covar_2.K().diagonal().mean()
        V[2] = gp.getParams()['lik'][0]**2
        V/=V.sum()
        self.Vconfounder = V
    
