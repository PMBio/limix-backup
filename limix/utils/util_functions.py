import sys
import scipy as SP
import h5py
import pdb

def vec(M):
    return M.reshape((M.size, 1), order = 'F')

def to_list(x):
    if x is None:               r = []
    elif type(x) is not list:   r = [x]
    else:                       r = x
    return r

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

