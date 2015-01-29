import sys
import scipy 
import pdb
import h5py
from utils import *

import sys
sys.path.insert(0,'../../')
from settings import *

import glob
import matplotlib.pylab as PLT

def plot_results(fn):
    identifier = fn.split('/')[-1].split('.')[0]
    f = h5py.File(fn,'r')

    LML = []
    UB  = []
    LB  = []
    
    fig = PLT.figure(1)
    keys = f.keys()
    for i,key in enumerate(keys):
        fig.add_subplot(4,3,3*i+1)
        PLT.title('Cfg[%s]'%key)
        PLT.imshow(f[key]['Cfg'][:],interpolation='nearest')
        PLT.xticks([]); PLT.yticks([]);
        
        fig.add_subplot(4,3,3*i+2)
        PLT.title('Cbg[%s]'%key)
        PLT.imshow(f[key]['Cbg'][:],interpolation='nearest')
        PLT.xticks([]); PLT.yticks([]);
        
        fig.add_subplot(4,3,3*i+3)
        PLT.title('Cn[%s]'%key)
        PLT.imshow(f[key]['Cn'][:],interpolation='nearest')
        PLT.xticks([]); PLT.yticks([]);

        LML.append(f[key]['LML'][:])
        UB.append(f[key]['UB'][:])
        LB.append(f[key]['LB'][:])
        
    PLT.savefig('figures/approximations/%s_covs.pdf'%identifier)
    PLT.close()
    
    fig = PLT.figure(2)
    PLT.bar(scipy.arange(len(LB)),LB,width=0.25,color='b')
    PLT.bar(scipy.arange(len(LML))+0.3,LML,width=0.25,color='g')
    PLT.bar(scipy.arange(len(UB))+0.6,UB,width=0.25,color='r')
    PLT.xticks(scipy.arange(len(UB))+0.5,keys)
    PLT.savefig('figures/approximations/%s_bound.pdf'%identifier)
    PLT.close()
    f.close()

if __name__ == "__main__":
    dataset_name = sys.argv[1]

    fns = glob.glob('out/approximation/*%s*'%(dataset_name))

    for fn in fns:
        plot_results(fn)
   
