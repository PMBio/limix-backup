import sys
import ipdb
import numpy.linalg as la 
import numpy as np
import scipy as sp


class GeneraterKron(object):
    def __init__(self,N=100,S=1000,R=1000,P=3,var_K1=0.6,var_K2=0.4,h2_1=0.9,h2_2=0.1,A=None, var_snps=None,h2=0.5):
        assert var_K1>=0, "var_K1 has to be greater or equal to zero"
        assert var_K2>=0, "var_K2 has to be greater or equal to zero"
        assert h2_1<=1, "h2_1 has to be smaller or equal to one"
        assert h2_1>=0, "h2_1 has to be larger or equal to 0"
        assert h2_2<=1, "h2_1 has to be smaller or equal to one"
        assert h2_2>=0, "h2_1 has to be larger or equal to 0"

        self.A = A
        self.N = N
        self.P = P
        self.R = R

        self.debug = True

        self.h2_1 = h2_1
        self.h2_2 = h2_2
        self.var_K1 = var_K1
        self.var_K2 = var_K2

        if var_snps is None:
            self.h2 = h2
        else:
            assert var_snps>=0.0
            self.var_snps = var_snps

        self.generateR1()
        self.generateR2()
        self.generateC1()
        self.generateC2()
        self.generate_snps(S)
        self.generate_phenotype()
    
    @property
    def h2(self):
        return self.var_snps/(self.var_total)

    @h2.setter
    def h2(self,h2):
        assert h2>=0.0
        assert h2<=1.0
        self.var_snps = h2/(1-h2) * (self.var_K1 + self.var_K2)        

    @property
    def var(self):
        return self.var_K1+self.var_K2
    
    @property
    def var_total_empirical(self):
        return (np.sqrt(self.var_snps) * self.Y_snps + self.Y_back).var()

    @property
    def var_empirical(self):
        return self.Y_back.var()

    @property
    def var_total(self):
        return self.var+self.var_snps

    @property
    def Y_back(self):
        return np.sqrt(self.var_K1) * self.Y1 + np.sqrt(self.var_K2) * self.Y2
    
    @property
    def Y(self):
        return np.sqrt(self.var_snps) * self.Y_snps + self.Y_back + self.mean

    def generate_phenotype(self):
        self.generate_snp_effects()
        self.generate_background()
        self.generate_mean()
        return self.Y

    def generate_mean(self):
        mean = np.random.randn(P)+np.random.normal()
        self.mean = mean[np.newaxis,:]
        return self.mean

    def generate_background(self):
        self.Y1 = generate_matrix_normal(self.R1,self.C1)
        self.Y2 = generate_matrix_normal(self.R2,self.C2)
        return self.Y1,self.Y2
    
    def generate_snp_effects(self):
        self.Y_snps = generate_snp_effects_kron(snps=self.snps,P=self.P,A=self.A,standardize=True)
        return self.Y_snps

    def Rrot(self):
        return rot_kron(self.R1,self.R2)

    def Crot(self):
        return rot_kron(self.C1,self.C2)

    def Yrot(self):
        res = self.Rrot().dot(self.Y).dot(self.Crot())
        return res

    def snps_rot(self):
        res = self.Rrot().dot(self.snps)
        return res

    def A_rot(self):
        if self.A is None:
            return self.Crot()
        else:
            return self.Crot().dot(self.A)
            

    @property
    def S(self):
        return self.snps.shape[1]
     
    def generate_snps(self,S):
        self.snps = np.random.randn(self.N,S)
        
    def generateR1(self):
        self.R1,self.X1 = generate_kernel(N=self.N, R=self.R, h2=self.h2_1)

    def generateR2(self):
        self.R2,self.X2 = generate_kernel(N=self.N, R=self.R, h2=self.h2_2)

    def generateC1(self):
        self.C1,self.A1 = generate_kernel(N=self.P, R=self.R, h2=self.h2_1)

    def generateC2(self):
        self.C2,self.A2 = generate_kernel(N=self.P, R=self.R, h2=self.h2_2)

    def covariance_vec1(self):
        return np.kron(self.C1,self.R1)
    
    def covariance_vec2(self):
        return np.kron(self.C2,self.R2)

    def K_vecY(self):
        return self.var_K1*self.covariance_vec1() + self.var_K2*self.covariance_vec2()

    def K_vecY_inv(self):
        cov = self.K_vecY()
        cov_inv = la.inv(cov)
        return cov_inv

    def KinvY_vec(self):
        cov = self.K_vecY()
        y = self.vecY()
        Ky = la.solve(cov,y)
        return Ky

    def YKY_vec(self):
        y = self.vecY()
        Ky = self.KinvY_vec
        return y.T.dot(Ky)

    def resKres_vec(self):
        res = self.residual_vec()
        K = self.K_vecY()
        Kres = la.solve(K,res)
        return res.T.dot(Kres)

    def logdet_K_vec(self):
        K = self.K_vecY()
        sign,logdet = la.slogdet(K)
        return logdet

    def logl_vec(self,reml=False):
        if reml:
            raise Exception("not implemented")
        yKy = self.resKres_vec()
        var_ml = yKy/(self.N*self.P)
        
        const = self.N*self.P *np.log(2.0*sp.pi)
        logdet = self.logdet_K_vec()
        dataterm = yKy/var_ml
        logl =  -0.5 * (const + logdet + dataterm)
        return logl

    def var_ml_vec(self):
        return self.resKres_vec()/(self.N*self.P)

    def XKY_vec(self):
        Ky = self.KinvY_vec()
        X = self.vecX()
        return X.T.dot(Ky)

    def beta_vec(self):
        XKX = self.XKX_vec()
        XKy = self.XKY_vec()
        beta = la.solve(XKX,XKy)
        return beta

    def predict_vec(self):
        X = self.vecX()
        beta = self.beta_vec()
        return X.dot(beta)

    def residual_vec(self):
        y = self.vecY()
        return y - self.predict_vec()

    def KX_vec(self):
        X=self.vecX()
        cov = self.K_vecY()
        KX = la.solve(cov,X)
        return KX

    def XKX_vec(self):
        KX = self.KX_vec()
        X = self.vecX()
        return X.T.dot(X)

    def vecY(self):
        return self.Y.flatten(order="C")
    
    def vecX(self):
        if self.A is None:
            return np.kron(np.eye(self.P),self.snps)
        else:
            return np.kron(self.A,self.snps)

    def d_XKX(self):
        raise Exception("not yet impl")

    def d_logl(self):
        raise Exception("not yet impl")

    def d_K(self):
        raise Exception("not yet impl")



def generate_kernel(N,R=100, h2=0.99999999):
    assert h2>=0.0
    assert h2<=1.0
    X = np.random.randn(N,R)#random effects
    XX = X.dot(X.T)
    XX /= XX.diagonal().mean()
    R = h2 * XX + (1-h2) * np.eye(N)
    return R, X

def generate_matrix_normal(R,C):
    N=R.shape[0]
    P=C.shape[0]
    sqrtR = la.cholesky(R)
    sqrtC = la.cholesky(C)
    Y = np.random.randn(N,P)
    Y = np.dot(sqrtR, Y)
    Y = np.dot(Y,sqrtC)
    return Y

def rot_kron(C1,C2):
    sqrtC2i = pexp(C2,exp=-0.5)
    C1_rot = sqrtC2i.T.dot(self.C1).dot(sqrtC2i)
    Crot = la.eigh(Crot)
    return Crot

def pexph(M,exp=0.5,eps=1e-8,symmetric=True,debug=False):
    s,u = la.eigh(M)
    i_nonz = s>=eps
    if i_nonz.any():
        s=s[i_nonz]
        u=u[:,i_nonz]
        res = u * (s**exp)[np.newaxis,:]
        if symmetric:
            res = res.dot(u.T)
    else:
        if symmetric:
            res = np.zeros_like(M)
        else:
            res = np.zeros((M.shape[0],0))
    if debug:
        M_ = pexph(M,exp=1.0)
        diff = np.absolute(M-M_).sum()
        print diff
        assert diff<1E-8
    return res

def pexp(M,exp=0.5, eps=1e-8,symmetric=True,debug=False):
    u,s,v = la.svd(M,full_matrices=False)
    i_nonz = s>=eps
    if i_nonz.any():
        s=s[i_nonz]
        u=u[:,i_nonz]
        v=v[i_nonz]
        res = u * (s**exp)[np.newaxis,:]
        if symmetric:
            res = res.dot(v)
    else:
        if symmetric:
            res = np.zeros_like(M)
        else:
            res = np.zeros((M.shape[0],0)) 
    if debug:
        M_ = pexp(M,exp=1.0)
        diff = np.absolute(M-M_).sum()
        print diff
        assert diff<1E-8
    return res

def generate_snp_effects_kron(snps,P=1,A=None, standardize=True, eps=1E-8):
    assert eps<=1.0
    assert eps>=0.0
    if A is None:
        A = np.eye(P)
    XX = (1.0-eps)*snps.dot(snps.T)+eps*np.eye(N)
    if standardize:
        XX/=XX.diagonal().mean()
    AA = (1.0-eps)*A.dot(A.T)+eps*np.eye(P)
    if standardize:
        AA/=AA.diagonal().mean()
    Y = generate_matrix_normal(R=XX,C=AA)
    return Y

def generate_snp_effects_primal(snps,P=1,A=None):
    if A is None:
        A = np.eye(P)
    beta = np.random.randn(snps.shape[1],A.shape[1])
    Y = snps.dot(beta.dot(A.T))
    return beta,Y

 

if __name__ == "__main__":

    plot = "plot" in sys.argv

    # generate data
    N=100
    S=5
    R=1000
    P=3
    var_K1=0.6
    var_K2=0.4
    h2_1=0.9
    h2_2=0.1

    h2 = 0.5

    A=None
    data = GeneraterKron(N=N,S=S,R=R,P=P,var_K1=var_K1,var_K2=var_K2,h2_1=h2_1,h2_2=h2_2,h2=h2)
    
    XKX = data.XKX_vec()
