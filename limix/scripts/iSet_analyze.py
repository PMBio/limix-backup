#! /usr/bin/env python
# Copyright(c) 2014, The mtSet developers (Francesco Paolo Casale, Barbara Rakitsch, Oliver Stegle)
# All rights reserved.

import time
import sys
import os
from limix.mtSet.iset import fit_iSet
from optparse import OptionParser
import numpy as np
import pandas as pd
import scipy as sp
import csv

from ..mtSet.core.read_utils import readNullModelFile
from ..mtSet.core.read_utils import readWindowsFile
from ..mtSet.core.read_utils import readCovarianceMatrixFile
from ..mtSet.core.read_utils import readCovariatesFile
from ..mtSet.core.read_utils import readPhenoFile
from ..mtSet.core import plink_reader

def entry_point():
    parser = OptionParser()
    parser.add_option("--bfile", dest='bfile', type=str, default=None)
    parser.add_option("--cfile", dest='cfile', type=str, default=None)
    parser.add_option("--pfile", dest='pfile', type=str, default=None)
    parser.add_option("--wfile", dest='wfile', type=str, default=None)
    parser.add_option("--ffile", dest='ffile', type=str, default=None)
    parser.add_option("--ifile", dest='ifile', type=str, default=None)
    parser.add_option("--resdir", dest='resdir', type=str, default='./')
    parser.add_option("--trait_idx",dest='trait_idx',type=str, default=None)

    # start window, end window and permutations
    parser.add_option("--minSnps", dest='minSnps', type=int, default=4)
    parser.add_option("--n_perms", type=int, default=10)
    parser.add_option("--start_wnd", dest='i0', type=int, default=None)
    parser.add_option("--end_wnd", dest='i1', type=int, default=None)
    parser.add_option("--factr", dest='factr', type=float, default=1e7)

    (options, args) = parser.parse_args()

    print('importing data')
    if options.cfile is None:
        import warnings
        cov = {'eval':None, 'evec':None}
        warnings.warn('warning: cfile not specifed, a one variance compoenent'+
                      ' model will be considered')
    else:
        cov = readCovarianceMatrixFile(options.cfile, readCov=False)

    Y = readPhenoFile(options.pfile,idx=options.trait_idx)

    wnds = readWindowsFile(options.wfile)

    bim = plink_reader.readBIM(options.bfile,usecols=(0,1,2,3))
    fam = plink_reader.readFAM(options.bfile,usecols=(0,1))
    chrom = bim[:, 0].astype(float)
    pos = bim[:, -1].astype(float)

    i0 = 1 if options.i0 is None else options.i0
    i1 = wnds.shape[0] if options.i1 is None else options.i1

    df = pd.DataFrame()
    df0 = pd.DataFrame()
    t0 = time.time()

    S_R = cov['eval']
    U_R = cov['evec']

    if options.ifile is None:
        strat = False
    else:
        strat = True
        print(".. loading indicator file %s " % options.ifile)
        Ie = np.asarray(pd.read_csv(options.ifile, index_col=0)).flatten()
        covs = sp.concatenate([U_R[:,-10:], sp.ones([U_R.shape[0], 1])], 1)

    res_dir = os.path.join(options.resdir,'test')

    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    n_digits = len(str(wnds.shape[0]))
    fname = str(i0).zfill(n_digits)
    fname+= '_'+str(i1).zfill(n_digits)
    resfile = os.path.join(res_dir, fname)

    for wnd_i in range(i0,i1):
        print(('.. window %d - (%d, %d-%d) - %d snps'%(wnd_i,int(wnds[wnd_i,1]),int(wnds[wnd_i,2]),int(wnds[wnd_i,3]),int(wnds[wnd_i,-1]))))
        if int(wnds[wnd_i,-1])<options.minSnps:
            print('SKIPPED: number of snps lower than minSnps')
            continue
        Xr = plink_reader.readBED(options.bfile, useMAFencoding=True, start = int(wnds[wnd_i,4]), nSNPs = int(wnds[wnd_i,5]), bim=bim , fam=fam)['snps']
        Xr = np.ascontiguousarray(Xr)
        Xr-= Xr.mean(0)
        Xr/= Xr.std(0)
        Xr/= np.sqrt(Xr.shape[1])

        if strat:
            _df, _df0 = fit_iSet(Y[:,[0]], Xr=Xr, covs=covs,
                                 n_perms=options.n_perms, Ie=Ie, strat=strat)
        else:
            _df, _df0 = fit_iSet(Y, U_R=U_R, S_R=S_R, Xr=Xr,
                                 n_perms=options.n_perms, strat=strat)

        _df.index = [wnd_i]
        _df.index.name = 'window'

        _df0.index = ['%d_%d' % (wnd_i, perm) for perm in range(_df0.shape[0])]
        _df0.index.name = 'window_perm'

        df = df.append(_df)
        df0 = df0.append(_df0)
    print 'Elapsed:', time.time()-t0

    df.to_csv(resfile + '.iSet.real')
    df0.to_csv(resfile + '.iSet.perm')
