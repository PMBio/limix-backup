from settings import CFG
import h5py
import pdb
import scipy 
import scipy.linalg 
import sys
import os
import matplotlib
matplotlib.use('agg')

import matplotlib.pylab as pltfig = plt.figure()

def run_experiments(dataset_name,seed):
    scipy.random.seed(seed)
    if dataset_name=='nfbc':
        # load NFBC data
        f = h5py.File(CFG['nfbc']['data'],'r')
        Kpop = f['Kpop'][:]
        f.close()
    elif dataset_name=='arab':
        # load arabidopsis
        maf = 0.05
        f = h5py.File(CFG['arab']['data'],'r')
        X = f['genotype']['matrix'][:]
        idx_maf = scipy.sum(X==1,axis=0)>maf*X.shape[0]
        X = X[:,idx_maf]
        # downsample SNPs to increase speed
        X = X[:,scipy.random.permutation(X.shape[1])[:10000]]
        X-= X.mean(0)
        X/= X.std(0)
        Kpop = scipy.dot(X,X.T)
        Kpop/= scipy.diag(Kpop).mean()
        f.close()

    # downsample samples to increase speed
    Kpop = Kpop[:1000][:,:1000]
    
    # simulate phenotype
    S,U   = scipy.linalg.eigh(Kpop)
    Sroot = scipy.sqrt(S)
    N     = Kpop.shape[0]
    I     = scipy.eye(N)
    Ysignal = scipy.random.randn(N,nReps)
    Ysignal = scipy.dot(U,(Sroot*Ysignal.T).T)    
    Ynoise  = scipy.random.randn(N,nReps)

    sample_sizes = [200,400,600,800,1000]
    logdet = scipy.zeros((nReps,len(sample_sizes)))
    sf     = scipy.zeros((nReps,len(sample_sizes)))

    
    for i_rep in range(nReps):
        # simulate phenotype
        sigma_g = scipy.random.rand()
        y = sigma_g*Ysignal[:,i_rep] + (1-sigma_g)*Ynoise[:,i_rep]
        K = sigma_g*Kpop + (1-sigma_g)*I
        idx = scipy.random.permutation(N)

        # compute squared form and log determinant
        for i_n,n in enumerate(sample_sizes):
            Kn = K[idx[:n]][:,idx[:n]]
            S,U = scipy.linalg.eigh(Kn)
            logdet[i_rep,i_n] = scipy.log(S).sum()
            sf[i_rep,i_n]     = scipy.dot(y[idx[:n]],scipy.linalg.solve(Kn,y[idx[:n]]))

    f = h5py.File(fn_out)
    f['sf']     = sf
    f['logdet'] = logdet
    f['n']      = sample_sizes
    f.close()
    

if __name__ == "__main__":
    nReps = 100
    dataset_name = sys.argv[1]
    seed         = int(sys.argv[2])

    fn_out = 'out/samplesize/%s_%d.hdf'%(dataset_name,seed)

    if not(os.path.exists(fn_out)) or 'recalc' in sys.argv:
        run_experiments(dataset_name,seed)

    # loading results
    f = h5py.File(fn_out,'r')
    sf     = f['sf'][:]
    logdet = f['logdet'][:]
    samplesizes = f['n'][:]
    f.close()

    # plotting
    ratio = scipy.absolute(logdet)/(scipy.absolute(sf) + scipy.absolute(logdet))

    pdb.set_trace()
    fig = plt.figure(figsize=(12,3))
    
    for i,samplesize in enumerate(samplesizes):
        fig.add_subplot(1,len(samplesizes),i+1)
        plt.title(samplesize)
        plt.hist(ratio[:,i],alpha=0.5,normed=True,range=(0.0,1.0),bins=10)
        
    plt.savefig('figures/samplesize/%s_%d.pdf'%(dataset_name,seed))
    pdb.set_trace()
            
