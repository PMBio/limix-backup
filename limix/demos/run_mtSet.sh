#!/bin/bash 
BFILE=./data/1000G_chr22/chrom22_subsample20_maf0.10 #specify here bed basename
CFILE=./out/chrom22
PFILE=./out/pheno
WFILE=./out/windows
NFILE=./out/null
WSIZE=30000
RESDIR=./out/results
OUTFILE=./out/final

# Simulating the phenotypes
./../../scripts/mtSet_simPheno --bfile $BFILE --cfile $CFILE --pfile $PFILE --chrom 22 --minPos 1640000  --maxPos 17550000

# Preprocessing and generation
./../../scripts/mtSet_preprocess --compute_covariance --bfile $BFILE --cfile $CFILE 

./../../scripts/mtSet_preprocess --precompute_windows --fit_null --bfile $BFILE --cfile $CFILE --pfile $PFILE --wfile $WFILE --nfile $NFILE --window_size $WSIZE --plot_windows

# Analysis
# test
./../../scripts/mtSet_analyze --bfile $BFILE --cfile $CFILE --pfile $PFILE --nfile $NFILE --wfile $WFILE --minSnps 4 --resdir $RESDIR --start_wnd 0 --end_wnd 100

#permutations
for i in `seq 0 10`;
do
./../../scripts/mtSet_analyze --bfile $BFILE --cfile $CFILE --pfile $PFILE --nfile $NFILE --wfile $WFILE --minSnps 4 --resdir $RESDIR --start_wnd 0 --end_wnd 100 --perm $i
done

#postprocess
./../../scripts/mtSet_postprocess --resdir $RESDIR --outfile $OUTFILE --manhattan_plot
