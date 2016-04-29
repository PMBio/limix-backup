# Copyright(c) 2014, The LIMIX developers (Christoph Lippert, Paolo Francesco Casale, Oliver Stegle)
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.
import numpy as np
import scipy as sp
import glob
import pdb, sys, pickle
import matplotlib.pylab as plt
from optparse import OptionParser
# from util import *
import os, shutil
import subprocess as s
#import covar
#import one_chr
np.random.seed(1)
#from recode import *

os.putenv('FastLmmUseAnyMklLib', '1')
plink_path = 'plink' # 'D:/users/lippert/code/plink.exe'
fastlmm_path = 'fastlmmc' # 'D:/users/lippert/Projects/FaSTLMM/CPP/x64/MKL-Release/FastLmmC.exe'


def get_column(filename, col, skiprows = 1, dtype = float):
    import pdb
    f = open(filename, 'r')
    results = []
    skipped = 0
    for line in f:
        if skipped < skiprows:
            skipped += 1
            continue
        data = sp.array(line.strip('\n').split('\t'))
        results.append(data[col])

    try:
        results = np.asarray(results, dtype = dtype)
    except ValueError:
        results = np.asarray(results, dtype = str)
        results = np.asarray(results[results != 'NA'], dtype = dtype)
    return results

#1) determine covariance matrix
#1.1) full RRM
#1.2) SNPs list
#1.2.1) subsampling (equally spaced)
#1.2.2) select
#1.2.2.1) linear regression - lambda
#1.2.2.2) linear regression - out of sample
#2) run fastlmm
#3) plot results

def load_pvals(filename, return_info = False, linreg = False):
    if linreg:
        i_pval = 4
    else:
        i_pval = 5
    if return_info:
        index = sp.array([i_pval, 1, 3])
    else:
        index = i_pval
    pvals = get_column(filename,index)
    return pvals

def extract_topN(N, selsnps_file, linreg_out):
    ordered_snps = get_column(linreg_out, 0, dtype = str)[:N]
    np.savetxt(selsnps_file, ordered_snps, fmt = '%s')

def run_select_topN(pheno_file,linreg_out,file_test=None,bfile_test = None,tfile_test = None,bfile_sim = None,file_sim = None,tfile_sim= None, out_dir='./tmp', covariates = None, excl_dist = None, excl_pos =
None, chr_only = None, quiet = False, refit_delta = False, command=None, Nmin=0, Nmax=1000, increment=100 ,Nsnps = None,recompute = False):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    selsnps_file =  os.path.join(out_dir, 'selectlist.txt')
    extract_topN( Nmax,selsnps_file, linreg_out )

    if Nsnps is None:
        Nsnps = sp.arange(Nmin,Nmax,increment)
    lambdas = sp.zeros(Nsnps.shape)
    for i in range(Nsnps.shape[0]):
        current_dir = 'N%d'% Nsnps[i]
        out_dir_current = os.path.join(out_dir, current_dir)
        if Nsnps[i]>0:
            run_fastlmm(pheno_file,file_test,bfile_test,tfile_test,bfile_sim,file_sim,tfile_sim, out_dir_current, covariates, excl_dist, excl_pos, chr_only, quiet, Nsnps[i],selsnps_file,refit_delta, command, True,recompute=recompute)
            out_file_current = os.path.join(out_dir_current, 'results_LMM.txt')
            pvals = get_column(out_file_current,5)
        else:
            run_linreg(pheno_file,file_test,bfile_test,tfile_test, out_dir_current, covariates, chr_only, quiet, command, True,recompute=recompute)
            out_file_current = os.path.join(out_dir_current, 'results_linreg.txt')
            pvals = get_column(out_file_current,4)
        lambdas[i] = estimate_lambda(pvals.flatten())
    return Nsnps, lambdas

def run_linreg(pheno_file,file_test=None,bfile_test = None,tfile_test = None, out_dir='./tmp', covariates = None, chr_only = None, quiet = False, command=None, run = True,recompute = False,**kw_args):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_file = os.path.join(out_dir, 'results_linreg.txt')

    if command is None:
        command = ('%s -pheno %s -verboseOut -out %s -linreg' %(fastlmm_path, pheno_file, out_file))

    #SNP files for testing
    if bfile_test is not None: #binary PED format
        command += ' -bfile %s' % bfile_test
    elif tfile_test is not None: #TPED format
        command += ' -tfile %s' % tfile_test
    elif file_test is not None: #PED format
        command += ' -file %s' % file_test
    else:
        pass

    if covariates != None:
        command += " -covar %s" % covariates

    out_log = os.path.join(out_dir,'results_linreg.log')

    if len(glob.glob(out_log)) == 1:
        f = open(out_log,'r')
        for line in f:
            commandold = line.strip('\n')
            if commandold == command:
                run  = False
    np.savetxt(out_log, sp.array([command]),'%s')
    if (run or recompute):
        if quiet:
            print('Running linear regression ...')
            print(('%s' % (command)))
            s.check_call(command, shell = True, stdout = fnull, stderr = fnull)
        else:
            s.check_call(command, shell = True)
    else:
        print('returning without running linear regression')
    return command

def run_fastlmm(pheno_file,file_test=None,bfile_test = None,tfile_test = None,bfile_sim = None,file_sim = None,tfile_sim= None, out_dir='./tmp', covariates = None, excl_dist = None, excl_pos = None, chr_only = None, quiet = False, N=None,selsnps_file = None,refit_delta = False, command=None,run = True,recompute=False,**kw_args):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_file = os.path.join(out_dir, 'results_LMM.txt')
    if command is None:
        command = ('%s -pheno %s -verboseOut -out %s' %(fastlmm_path, pheno_file, out_file))
    #SNP files for testing
    if bfile_test is not None: #binary PED format
        command += ' -bfile %s' % bfile_test
    elif tfile_test is not None: #TPED format
        command += ' -tfile %s' % tfile_test
    elif file_test is not None: #PED format
        command += ' -file %s' % file_test
    else:
        pass
    #SNP files for RRM
    if bfile_sim is not None: #binary PED format
        command += ' -bfilesim %s' % bfile_sim
    elif tfile_sim is not None: #TPED format
        command += ' -tfilesim %s' % tfile_sim
    elif file_sim is not None: #PED format
        command += ' -filesim %s' % file_sim
    else:
        pass
    if covariates != None:
        command += " -covar %s" % covariates

    if N is not None:
        command+= " -extractSimTopK %s %d" %(selsnps_file,N)

    if excl_dist != None:
        command += ' -excludebygeneticdistance %f' % (excl_dist)
    elif excl_pos != None:
        command += ' -excludebyposition %d' %  (excl_pos)
    out_log = os.path.join(out_dir,'results_LMM.log')

    if len(glob.glob(out_log)) == 1:
        f = open(out_log,'r')
        for line in f:
            commandold = line.strip('\n')
            if commandold == command:
                run  = False
    np.savetxt(out_log, sp.array([command]),'%s')
    if (run or recompute):
        if quiet:
            print('Running LMM ...')
            print(command)
            s.check_call(command, shell = True, stdout = fnull, stderr = fnull)
        else:
            s.check_call(command, shell = True)
    else:
        print('returning without running LMM')
    return command



def get_options(opt = None):
    usage = 'usage: %prog [options] snps_file pheno_file'
    parser = OptionParser(usage = usage)
    parser.add_option('-f', '--file', dest = 'file_test', action = 'store', type = 'string', help = 'Plink PED fileset', default = None)
    parser.add_option('-b', '--bfile', dest = 'bfile_test', action = 'store', type = 'string', help = 'Plink binary fileset', default = None)
    parser.add_option('-t', '--tfile', dest = 'tfile_test', action = 'store', type = 'string', help = 'Plink TPED fileset', default = None)
    parser.add_option('-K', '--filesim', dest = 'file_sim', action = 'store', type = 'string', help = 'Plink PED fileset', default = None)
    parser.add_option('-L', '--bfilesim', dest = 'bfile_sim', action = 'store', type = 'string', help = 'Plink binary fileset', default = None)
    parser.add_option('-Q', '--tfilesim', dest = 'tfile_sim', action = 'store', type = 'string', help = 'Plink TPED fileset', default = None)
    parser.add_option('-N', '--n-snps', dest = 'N', action = 'store', type = 'int', help = 'number of SNPs to use to estimate Kinship (realized relationship) matrix (default all)', default = None)
    parser.add_option('-o', '--out', dest = 'out_dir', action = 'store', type = 'string', help = 'Output directory (default ./tmp)', default = './tmp')
    parser.add_option('-d', '--delimiter', dest = 'delimiter', action = 'store', type = 'string', help = 'delimiter between fields of the phenotype file', default = '\t')
    parser.add_option('-m', '--pheno-missing-value', dest = 'missing', action = 'store', type = 'float', help = 'missing value for the phenotype file (default -9)', default = -9.0)
    parser.add_option('-q', '--quiet', dest = 'quiet', action = 'store_true', help = 'show output from calls', default = False)
    parser.add_option('-C', '--covariates_file', dest = 'covariates', action = 'store', type = 'string', help = 'covariates', default = None)
    parser.add_option('-D', '--exclude_distance', dest = 'excl_dist', action = 'store', type = 'float', help = 'exclude by distance (in cM)', default = None)
    parser.add_option('-P', '--exclude_position', dest = 'excl_pos', action = 'store', type = 'int', help = 'exclude by position (in bp)', default = None)
    parser.add_option('-s', '--spaced', dest = 'spaced', action = 'store_true',  help = 'equally spaced regressors')
    parser.add_option('-T', '--testing', dest = 'testing', action = 'store_true',  help = 'only run the final LMM')
    parser.add_option('-R', '--one_chr', dest = 'chr_only', action = 'store_true',  help = 'test on one chr and build kernel on the rest')
    parser.add_option('-p', '--pheno', dest = 'pheno_file', action = 'store',  type = 'string', help = 'phenotype file')
    if opt is not None:
        (options, args) = parser.parse_args(opt)
    else:
        (options, args) = parser.parse_args()
    return options, args

if __name__ == '__main__':
    (options, args) = get_options()

    #command_LMM = run_fastlmm(pheno_file = options.pheno_file,file_test=options.file_test,bfile_test = options.bfile_test,tfile_test = options.tfile_test,bfile_sim = options.bfile_sim,file_sim = options.file_sim,tfile_sim= options.tfile_sim, out_dir=options.out_dir, covariates = options.covariates, excl_dist = options.excl_dist, excl_pos = options.excl_pos, chr_only = options.chr_only, testing = False, quiet = False, N=None,refit_delta = False)

#    command_linreg = run_linreg(pheno_file = options.pheno_file, file_test=options.file_test, bfile_test = options.bfile_test, tfile_test = options.tfile_test, out_dir=options.out_dir, covariates = options.covariates, chr_only = options.chr_only, testing = False, quiet = False)

    linreg_out = os.path.join(options.out_dir, 'results_linreg.txt')
    pvals = load_pvals(linreg_out,True,True )



    #Nsnps0,lambdas0 = run_select_topN(pheno_file= options.pheno_file, linreg_out=linreg_out,file_test=options.file_test,bfile_test = options.bfile_test,tfile_test = options.tfile_test,bfile_sim = options.bfile_sim,file_sim = options.file_sim, tfile_sim = options.tfile_sim, out_dir=options.out_dir, covariates = options.covariates, excl_dist = options.excl_dist, excl_pos = options.excl_pos, chr_only = options.chr_only, quiet = False, refit_delta = False, command=None, Nmin=0, Nmax=100, increment=10 )

    #Nsnps = sp.arange(20)

    #Nsnps,lambdas = run_select_topN(pheno_file= options.pheno_file, linreg_out=linreg_out,file_test=options.file_test,bfile_test = options.bfile_test,tfile_test = options.tfile_test,bfile_sim = options.bfile_sim,file_sim = options.file_sim, tfile_sim = options.tfile_sim, out_dir=options.out_dir, covariates = options.covariates, excl_dist = options.excl_dist, excl_pos = options.excl_pos, chr_only = options.chr_only, quiet = False, refit_delta = False, command=None, Nmin=0, Nmax=100, increment=10 ,Nsnps = Nsnps)
