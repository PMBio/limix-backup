#!/bin/bash
BFILE=./data/chrom22_subsample20_maf0.10 #specify here bed basename
CFILE=./out/chrom22
PFILE=./out/pheno
WFILE=./out/windows
NFILE=./out/null
WSIZE=30000
RESDIR=./out/results
OUTFILE=./out/final

# download the data
if [ ! -e data ]; then
    mkdir data
fi
if [ ! -e $BFILE.bed ]; then
    wget http://www.ebi.ac.uk/~casale/mtSet_demo/chrom22_subsample20_maf0.10.bed -P data
fi
if [ ! -e $BFILE.bim ]; then
    wget http://www.ebi.ac.uk/~casale/mtSet_demo/chrom22_subsample20_maf0.10.bim -P data
fi
if [ ! -e $BFILE.fam ]; then
    wget http://www.ebi.ac.uk/~casale/mtSet_demo/chrom22_subsample20_maf0.10.fam -P data
fi

# Compute covariance matrix
mtSet_preprocess --compute_covariance --bfile $BFILE --cfile $CFILE

# Simulating the phenotypes
mtSet_simPheno --bfile $BFILE --cfile $CFILE --pfile $PFILE --chrom 22 --minPos 1640000  --maxPos 17550000

# Precompute windows and fit null
mtSet_preprocess --precompute_windows --fit_null --bfile $BFILE --cfile $CFILE --pfile $PFILE --wfile $WFILE --nfile $NFILE --window_size $WSIZE --plot_windows

# Analysis
# test
mtSet_analyze --bfile $BFILE --cfile $CFILE --pfile $PFILE --nfile $NFILE --wfile $WFILE --minSnps 4 --resdir $RESDIR --start_wnd 0 --end_wnd 100

#permutations
for i in `seq 0 10`;
do
    mtSet_analyze --bfile $BFILE --cfile $CFILE --pfile $PFILE --nfile $NFILE --wfile $WFILE --minSnps 4 --resdir $RESDIR --start_wnd 0 --end_wnd 100 --perm $i
done

#postprocess
mtSet_postprocess --resdir $RESDIR --outfile $OUTFILE --manhattan_plot
