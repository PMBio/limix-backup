#!/bin/bash 
BFILE=./../data/1000G_chr22/chrom22_subsample20_maf0.10 #specify here bed basename
CFILE=./out/chrom22
PFILE=./out/pheno
WFILE=./out/windows
NFILE=./out/nullPCs
WSIZE=30000
RESDIR=./out/resultsPCs
OUTFILE=./out/finalPCs
FFILE=./out/pcs.txt

# Preprocessing and generation
./../mtSet/bin/mtSet_preprocess --compute_covariance --bfile $BFILE --cfile $CFILE 
./../mtSet/bin/mtSet_simPheno --bfile $BFILE --cfile $CFILE --pfile $PFILE --chrom 22 --minPos 1640000  --maxPos 17550000

./../mtSet/bin/mtSet_preprocess --precompute_windows --bfile $BFILE --cfile $CFILE --pfile $PFILE --wfile $WFILE --window_size $WSIZE --plot_windows --ffile $FFILE --compute_PCs 2

./../mtSet/bin/mtSet_preprocess --fit_null --bfile $BFILE --pfile $PFILE --nfile $NFILE  --ffile $FFILE 

# Analysis
# test
./../mtSet/bin/mtSet_analyze --bfile $BFILE  --pfile $PFILE --nfile $NFILE --wfile $WFILE --minSnps 4 --resdir $RESDIR --start_wnd 0 --end_wnd 100 --ffile $FFILE 


#permutations
for i in `seq 0 10`;
do
./../mtSet/bin/mtSet_analyze --bfile $BFILE  --pfile $PFILE --nfile $NFILE --wfile $WFILE --minSnps 4 --resdir $RESDIR --start_wnd 0 --end_wnd 100 --perm $i --ffile $FFILE 

done

#postprocess
./../mtSet/bin/mtSet_postprocess --resdir $RESDIR --outfile $OUTFILE --manhattan_plot
