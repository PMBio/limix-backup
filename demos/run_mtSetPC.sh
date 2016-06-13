#!/bin/bash
BFILE=./data/chrom22_subsample20_maf0.10 #specify here bed basename
CFILE=./out/chrom22
PFILE=./out/pheno
WFILE=./out/windows
NFILE=./out/nullPCs
WSIZE=30000
RESDIR=./out/resultsPCs
OUTFILE=./out/finalPCs
FFILE=./out/pcs.txt

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

# Generate phenotype
mtSet_simPheno --bfile $BFILE --cfile $CFILE --pfile $PFILE --chrom 22 --minPos 1640000  --maxPos 17550000

# precompute windows
mtSet_preprocess --precompute_windows --bfile $BFILE --cfile $CFILE --pfile $PFILE --wfile $WFILE --window_size $WSIZE --plot_windows --ffile $FFILE --compute_PCs 2

#fit null model
mtSet_preprocess --fit_null --bfile $BFILE --pfile $PFILE --nfile $NFILE  --ffile $FFILE

# Analysis
# test
mtSet_analyze --bfile $BFILE  --pfile $PFILE --nfile $NFILE --wfile $WFILE --minSnps 4 --resdir $RESDIR --start_wnd 0 --end_wnd 100 --ffile $FFILE

#permutations
for i in `seq 0 10`;
do
    mtSet_analyze --bfile $BFILE  --pfile $PFILE --nfile $NFILE --wfile $WFILE --minSnps 4 --resdir $RESDIR --start_wnd 0 --end_wnd 100 --perm $i --ffile $FFILE
done

#postprocess
mtSet_postprocess --resdir $RESDIR --outfile $OUTFILE --manhattan_plot
