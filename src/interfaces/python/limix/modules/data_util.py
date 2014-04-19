import scipy as SP

try:
    #see if fastlmm is in the path for the fast C-based parser
    import fastlmm.pyplink.wrap_plink_parser as parser  
    cparser=True
except:
    cparser=False
    pass

def estCumPos(pos,chrom,offset = 20000000):
    '''
    compute the cumulative position of each variant given the position and the chromosome
    Also return the starting cumulativeposition of each chromosome

    Args:
        pos:        scipy.array of basepair positions (on the chromosome)
        chrom:      scipy.array of chromosomes
        offset:     offset between chromosomes for cumulative position (default 20000000 bp)
    
    Returns:
        cum_pos:    scipy.array of cumulative positions
        chrom_pos:  scipy.array of starting cumulative positions for each chromosme
    '''
    chromvals = SP.unique(chrom)#SP.unique is always sorted
    chrom_pos=SP.zeros_like(chromvals)#get the starting position of each Chrom
    cum_pos = SP.zeros_like(pos)#get the cum_pos of each variant.
    maxpos_cum=0
    for i,mychrom in enumerate(chromvals):
        chrom_pos[i] = maxpos_cum
        i_chr=chrom==mychrom
        maxpos = pos[i_chr].max()+offset
        maxpos_cum+=maxpos
        cum_pos[i_chr]=chrom_pos[i]+pos[i_chr]
    return cum_pos,chrom_pos
        
        
def imputeMissing(X, center=True, unit=True, betaNotUnitVariance=False, betaA=1.0, betaB=1.0):
    '''
    fill in missing values in the SNP matrix by the mean value
    optionally center the data and unit-variance it

    Args:
        X:      scipy.array of SNP values. If dtype=='int8' the missing values are -9, 
                otherwise the missing values are scipy.nan
        center: Boolean indicator if data should be mean centered
                Not supported in C-based parser
        unit:   Boolean indicator if data should be normalized to have unit variance
                Not supported in C-based parser
        betaNotUnitVariance:    use Beta(betaA,betaB) standardization instead of unit variance 
                                (only with C-based parser) (default: False)
        betaA:  shape parameter for Beta(betaA,betaB) standardization (only with C-based parser)
        betaB:  scale parameter for Beta(betaA,betaB) standardization (only with C-based parser)
    
    Returns:
        X:      scipy.array of standardized SNPs with scipy.float64 values
    '''
    typeX=X.dtype
    if typeX!=SP.int8:
        iNanX = X!=X
    else:
        iNanX = X==-9
    if iNanX.any() or betaNotUnitVariance:
        if cparser:
            print "using C-based imputer"
            if X.flags["C_CONTIGUOUS"] or typeX!=SP.float32:
                X = SP.array(X, order="F", dtype=SP.float32)
                if typeX==SP.int8:
                    X[iNanX]=SP.nan
                parser.standardize(X,betaNotUnitVariance=betaNotUnitVariance,betaA=betaA,betaB=betaB)
                X=SP.array(X,dtype=SP.float64)
            else:
                parser.standardize(X,betaNotUnitVariance=betaNotUnitVariance,betaA=betaA,betaB=betaB)
            X=SP.array(X,dtype=SP.float64)
        else:
            if betaNotUnitVariance:
                raise NotImplementedError("Beta(betaA,betaB) standardization only in C-based parser, but not found")
            nObsX = (~iNanX).sum(0)
            if typeX!=SP.float64:
                X=SP.array(X,dtype=SP.float64)
            X[iNanX] = 0.0
            sumX = (X).sum(0)                
            meanX = sumX/nObsX
            if center:
                X-=meanX
                X[iNanX] = 0.0
                X_=X
            else:
                mean=SP.tile(meanX,(X.shape[0],1))
                X[iNanX]=mean[iNanX]
                X_=X-mean
            if unit:
                stdX = SP.sqrt((X_*X_).sum(0)/nObsX)
                stdX[stdX==0.0]=1.0
                X/=stdX
    else:
        if X.dtype!=SP.float64:
            X=SP.array(X,dtype=SP.float64)
        if center:
            X-= X.mean(axis=0)
        if unit:
            stdX= X.std(axis=0)
            stdX[stdX==0.0]=1.0
            X/=stdX
    return X