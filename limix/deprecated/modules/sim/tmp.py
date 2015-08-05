import h5py
import numpy as np

# filepath = '/Users/horta/workspace/1000G_majors.hdf5'
filepath = '/panfs/nobackup/research/stegle/users/casale'+\
           '/mksum/1000G/data/1000G_majors.hdf5'

with h5py.File(filepath, 'r+') as f:
    for k in f['genotypes'].keys():
        print k
        if 'chrom' not in k:
            continue
        g = f['genotypes'][k]
        if 'matrix_inds_by_snps' in g:
            continue
        Xt = g['matrix'][:]
        g.create_dataset('matrix_inds_by_snps', data=Xt.T)
