import h5py
import numpy as np

filepath = '/Users/horta/workspace/1000G_majors.hdf5'
# filepath = '/panfs/nobackup/research/stegle/users/casale'+\
#            '/mksum/1000G/data/1000G_majors.hdf5'

with h5py.File(filepath, 'r+') as f:
    for k in f['genotypes'].keys():
        print k
        if 'chrom' not in k:
            continue
        g = f['genotypes'][k]
        del g["matrix_inds_by_snps"]
        del g["matrix_snps_by_inds"]
        
        # if 'matrix_inds_by_snps_c' in g:
        #     del g["matrix_inds_by_snps_c"]

        # X = g['matrix_inds_by_snps'][:]
        # g.create_dataset('matrix_inds_by_snps_c', data=X,
        #                  chunks=(1, X.shape[1]),
        #                  compression="lzf")

        # X = g['matrix_snps_by_inds'][:]
        # g.create_dataset('matrix_snps_by_inds_c', data=X,
        #                  chunks=(1, X.shape[1]),
        #                  compression="lzf")
