import scipy 
import scipy.linalg
import h5py


def sim_psd_matrix(X=None,N=None,n_dim=None,jitter=1e-3):
    """
    simulate positive definite kernel
    """
    if X==None:
        X = scipy.random.randn(N,n_dim)
    else:
        N = X.shape[0]
        n_dim = X.shape[1]

    K = scipy.dot(X,X.T)
    K/= scipy.diag(K).mean()
    K+= jitter*scipy.eye(N)
 
    return K

def sim_psd_matrix(X=None,N=None,n_dim=None,jitter=1e-3):
    """
    simulate positive definite kernel
    """
    if X==None:
        X = scipy.random.randn(N,n_dim)
    else:
        N = X.shape[0]
        n_dim = X.shape[1]

    K = scipy.dot(X,X.T)
    K/= scipy.diag(K).mean()
    K+= jitter*scipy.eye(N)

    return K

def sim_kronecker(C,R):
    S_c,U_c = scipy.linalg.eigh(C+1E-6*scipy.eye(C.shape[0]))
    S_r,U_r = scipy.linalg.eigh(R+1E-6*scipy.eye(R.shape[0]))
    US_c = scipy.sqrt(S_c) * U_c
    US_r = scipy.sqrt(S_r) * U_r
    # kron(US_c,US_r) vec(Y) = vec(US_r.T*Y*US_c)
    Y = scipy.random.randn(R.shape[0],C.shape[0])
    Y = scipy.dot(US_r,scipy.dot(Y,US_c.T))
    return Y

def load_arabidopsis(fn,maf=0.05,debug=False):
    f = h5py.File(fn,'r')
    X = 1.*f['genotype']['matrix'][:]
    if debug: X = X[:200]
    idx_maf = scipy.sum(X==1,axis=0)>maf*X.shape[0]
    X = X[:,idx_maf]
    X-= X.mean(0)
    X/= X.std(0)
    chrom = f['genotype']['col_header']['chrom'][:][idx_maf]
    pos   = f['genotype']['col_header']['pos'][:][idx_maf]
    f.close()
    return X,chrom,pos
