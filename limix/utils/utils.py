import sys
import scipy as SP
import h5py
import pdb

def smartAppend(table,name,value):
    """
    helper function for apppending in a dictionary  
    """ 
    if name not in table.keys():
        table[name] = []
    table[name].append(value)

def smartSum(x,key,value):
    """ create a new page in x if key is not a page of x
        otherwise add value to x[key] """
    if key not in x.keys():
        x[key] = value
    else:   x[key]+=value

def dumpDictHdf5(RV,o):
    """ Dump a dictionary where each page is a list or an array """
    for key in RV.keys():
        o.create_dataset(name=key,data=SP.array(RV[key]),chunks=True,compression='gzip')

def smartDumpDictHdf5(RV,o):
    """ Dump a dictionary where each page is a list or an array or still a dictionary (in this case, it iterates)"""
    for key in RV.keys():
        if type(RV[key])==dict:
            g = o.create_group(key)
            smartDumpDictHdf5(RV[key],g)
        else:
            o.create_dataset(name=key,data=SP.array(RV[key]),chunks=True,compression='gzip')

def getCumPos(chrom,pos):
    """
    getCumulativePosition
    """
    n_chroms = int(chrom.max())
    x = 0
    for chrom_i in range(1,n_chroms+1):
        I = chrom==chrom_i

        if I.any():
            pos[I]+=x
            x=pos[I].max()
    return pos

def getChromBounds(chrom,posCum):
    """
    getChromBounds
    """
    n_chroms = int(chrom.max())
    chrom_bounds = []
    for chrom_i in range(2,n_chroms+1):
        I1 = chrom==chrom_i
        I0 = chrom==chrom_i-1
        _cb = 0.5*(posCum[I0].max()+posCum[I1].min())
        chrom_bounds.append(_cb)
    return chrom_bounds

