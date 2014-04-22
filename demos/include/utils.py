import scipy as SP
# import limix
import sys
sys.path.append('./../../build/release.darwin/interfaces/python')
import limix.modules.data as DATA

def getPosNew(data):
    """
    get Fixed position
    """
    pos = data.geno['col_header']['pos'][:]
    chrom= data.geno['col_header']['chrom'][:]
    n_chroms = chrom.max()
    pos_new = []
    for chrom_i in range(1,n_chroms+1):
        I = chrom==chrom_i
        _pos = pos[I]
        for i in range(1,_pos.shape[0]):
            if not _pos[i]>_pos[i-1]:
                _pos[i:]=_pos[i:]+_pos[i-1]
        pos_new.append(_pos)
    pos_new = SP.concatenate(pos_new)
    return pos_new

def getCumPos(data):
    """
    getCumulativePosition
    """
    pos = getPosNew(data)
    chrom= data.geno['col_header']['chrom'][:]
    n_chroms = int(chrom.max())
    x = 0
    for chrom_i in range(1,n_chroms+1):
        I = chrom==chrom_i
        pos[I]+=x
        x=pos[I].max()
    return pos

def getChromBounds(data):
    """
    getChromBounds
    """
    chrom= data.geno['col_header']['chrom'][:]
    posCum = getCumPos(data)
    n_chroms = int(chrom.max())
    chrom_bounds = []
    for chrom_i in range(2,n_chroms+1):
        I1 = chrom==chrom_i
        I0 = chrom==chrom_i-1
        _cb = 0.5*(posCum[I0].max()+posCum[I1].min())
        chrom_bounds.append(_cb)
    return chrom_bounds
