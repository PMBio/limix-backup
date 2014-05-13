import scipy as SP
import pandas as pd


try:
    #see if fastlmm is in the path for the fast C-based parser
    import fastlmm.pyplink.wrap_plink_parser as parser  
    cparser=True
except:
    cparser=False
    pass

def estCumPos(position,offset=1000000,chrom_len=None):
    '''
    compute the cumulative position of each variant given the position and the chromosome
    Also return the starting cumulativeposition of each chromosome

    Args:
        position:   pandas DataFrame of basepair positions (key='pos') and chromosome values (key='chrom')
                    The DataFrame will be updated with field 'pos_cum'
        offset:     offset between chromosomes for cumulative position (default 20000000 bp)
    
    Returns:
        chrom_pos:  numpy.array of starting cumulative positions for each chromosme
    '''
    chromvals = SP.unique(position['chrom'])#SP.unique is always sorted
    chrom_pos_cum=SP.zeros_like(chromvals)#get the starting position of each Chrom
    pos_cum=SP.zeros_like(position.shape[0])
    if not 'pos_cum' in position:
        position["pos_cum"]=SP.zeros_like(position['pos'])#get the cum_pos of each variant.
    pos_cum=position['pos_cum'].values
    maxpos_cum=0
    for i,mychrom in enumerate(chromvals):
        chrom_pos_cum[i] = maxpos_cum
        i_chr=position['chrom']==mychrom
        if chrom_len is None:
            maxpos = position['pos'][i_chr].max()+offset
        else:
            maxpos = chrom_len[i]+offset
        pos_cum[i_chr.values]=maxpos_cum+position.loc[i_chr,'pos']
        maxpos_cum+=maxpos      
    
    return chrom_pos_cum
        
        
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
        if cparser and center and (unit or betaNotUnitVariance):
            print "using C-based imputer"
            if X.flags["C_CONTIGUOUS"] and typeX==SP.float32:
                parser.standardizefloatCAAA(X,betaNotUnitVariance=betaNotUnitVariance,betaA=betaA,betaB=betaB)
                X=SP.array(X,dtype=SP.float64)
            elif X.flags["C_CONTIGUOUS"] and typeX==SP.float64:
                parser.standardizedoubleCAAA(X,betaNotUnitVariance=betaNotUnitVariance,betaA=betaA,betaB=betaB)
            elif X.flags["F_CONTIGUOUS"] and typeX==SP.float32:
                parser.standardizefloatFAAA(X,betaNotUnitVariance=betaNotUnitVariance,betaA=betaA,betaB=betaB)
                X=SP.array(X,dtype=SP.float64)
            elif X.flags["F_CONTIGUOUS"] and typeX==SP.float64:
                parser.standardizedoubleFAAA(X,betaNotUnitVariance=betaNotUnitVariance,betaA=betaA,betaB=betaB)
            else:
                X=SP.array(X,order="F",dtype=SP.float64)
                X[iNanX]=SP.nan
                parser.standardizedoubleFAAA(X,betaNotUnitVariance=betaNotUnitVariance,betaA=betaA,betaB=betaB)
        elif betaNotUnitVariance:
                raise NotImplementedError("Beta(betaA,betaB) standardization only in C-based parser, but not found")
        else:
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


def merge_indices(indices,header=None,join="inner"):
    """
    returns a merged index

    Args:
        indices:    list of indices (e.g. individual IDs)
        header:     list with name of each element of indices (e.g. ["pheno","geno","covars"])
        join:       type of join to perform (standard is "inner"

    Returns:
        pandas DataFrame with merged indices
    """
    indexpd = []
    for i, index in enumerate(indices):
        if header is None:
            header_=[i]
        else:
            header_=[header[i]]
        indexpd.append(pd.DataFrame(data=SP.arange(len(index)),index=index,columns=header_) )
    ret = pd.concat(objs=indexpd, axis=1, join=join)
    return ret

if __name__ == "__main__":
    lists=[["a","b"],["a","c","b"],["d","a","b"]]
    header = [["bl"],["n"],["s"]]
    merge=merge_indices(lists, header=None, join="outer")
    