# Copyright(c) 2014, The LIMIX developers (Christoph Lippert, Paolo Francesco Casale, Oliver Stegle)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pdb
import os
import numpy as SP
import csv

def deduce_delimiter(filename):
    with open(filename, 'r') as file:
        dialect = csv.Sniffer().sniff(file.read())
        return dialect.delimiter


def which(vec):
    for i in range(len(vec)):
        if (vec[i]):
            return(i)
    return(-1)


def readPED(basefilename, delimiter = ' ',missing = '0',standardize = True, pheno = None):
    '''
    read [basefilename].ped and [basefilename].map
    optionally standardize the SNPs and mean impute missing SNPs
    --------------------------------------------------------------------------
    Input:
    basefilename    : string of the basename of [basename].ped and [basename].map
    delimiter       : string (default ' ' space)
    missing         : string indicating a missing genotype (default '0')
    standardize     : boolean
                        True    : mean impute, zero-mean and unit variance the data
                        False   : output the data in 0,1,2 with NaN values
    --------------------------------------------------------------------------
    Output dictionary:
    'rs'     : [S] array rs-numbers, 
    'pos'    : [S*3] array of positions [chromosome, genetic dist, basepair dist],
    'snps'   : [N*S] array of snps-data,
    'iid'    : [N*2] array of family IDs and individual IDs
    --------------------------------------------------------------------------    
    '''
    pedfile = basefilename+".ped"
    mapfile = basefilename+".map"
    map = SP.loadtxt(mapfile,dtype = 'str')
    
    rs = map[:,1]
    pos = SP.array(map[:,(0,2,3)],dtype = 'float')
    map = None
    
    ped = SP.loadtxt(pedfile,dtype = 'str')
    iid = ped[:,0:2]
    snpsstr = ped[:,6::]
    inan=snpsstr==missing
    snps = SP.zeros((snpsstr.shape[0],snpsstr.shape[1]/2))
    if standardize:
        for i in range(snpsstr.shape[1]/2):
            snps[inan[:,2*i],i]=0
            vals=snpsstr[~inan[:,2*i],2*i:2*(i+1)]
            snps[~inan[:,2*i],i]+=(vals==vals[0,0]).sum(1)
            snps[~inan[:,2*i],i]-=snps[~inan[:,2*i],i].mean()
            snps[~inan[:,2*i],i]/=snps[~inan[:,2*i],i].std()
    else:
        for i in range(snpsstr.shape[1]/2):
            snps[inan[:,2*i],i]=SP.nan
            vals=snpsstr[~inan[:,2*i],2*i:2*(i+1)]
            snps[~inan[:,2*i],i]+=(vals==vals[0,0]).sum(1)
    if pheno is not None:
        #TODO: sort and filter SNPs according to pheno
        pass
    ret = {
           'rs'     :rs,
           'pos'    :pos,
           'snps'   :snps,
           'iid'    :iid
           }
    return ret

def readRAW(basefilename, delimiter = ' ',missing = '0',standardize = True, pheno = None):
    '''
    read [basefilename].raw
    optionally standardize the SNPs and mean impute missing SNPs
    --------------------------------------------------------------------------
    Input:
    basefilename    : string of the basename of [basename].ped and [basename].map
    delimiter       : string (default ' ' space)
    missing         : string indicating a missing genotype (default '0')
    standardize     : boolean
                        True    : mean impute, zero-mean and unit variance the data
                        False   : output the data in 0,1,2 with NaN values
    --------------------------------------------------------------------------
    Output dictionary:
    'rs'     : [S] array rs-numbers,
    'pos'    : [S*3] array of positions [chromosome, genetic dist, basepair dist],
    'snps'   : [N*S] array of snps-data,
    'iid'    : [N*2] array of family IDs and individual IDs
    --------------------------------------------------------------------------    
    '''
    rawfile = basefilename+".raw"
    #mapfile = basefilename+".map"
    #map = SP.loadtxt(mapfile,dtype = 'str')
    
    #rs = map[:,1]
    #pos = SP.array(map[:,(0,2,3)],dtype = 'float')
    #map = None
    import pdb
    #pdb.set_trace()
    raw = SP.loadtxt(rawfile,dtype = 'str')
    iid = raw[:,0:2]
    snpsstr = raw[:,6::]
    inan=snpsstr==missing
    snps = SP.zeros((snpsstr.shape[0],snpsstr.shape[1]/2))
    if standardize:
        for i in range(snpsstr.shape[1]/2):
            raw[inan[:,2*i],i]=0
            vals=snpsstr[~inan[:,2*i],2*i:2*(i+1)]
            snps[~inan[:,2*i],i]+=(vals==vals[0,0]).sum(1)
            snps[~inan[:,2*i],i]-=snps[~inan[:,2*i],i].mean()
            snps[~inan[:,2*i],i]/=snps[~inan[:,2*i],i].std()
    else:
        for i in range(snpsstr.shape[1]/2):
            snps[inan[:,2*i],i]=SP.nan
            vals=snpsstr[~inan[:,2*i],2*i:2*(i+1)]
            snps[~inan[:,2*i],i]+=(vals==vals[0,0]).sum(1)
    if pheno is not None:
        #TODO: sort and filter SNPs according to pheno
        pass
    ret = {
           'rs'     :rs,
           'pos'    :pos,
           'snps'   :snps,
           'iid'    :iid
           }
    return ret

def loadPhen(filename, missing ='-9', pheno = None):
    '''
    load a phenotype or covariate file. Covariates have the same file format.
    --------------------------------------------------------------------------
    Input:
    filename        : string of the filename
    missing         : string indicating a missing phenotype (default '-9')
    --------------------------------------------------------------------------
    Output dictionary:
    'header' : [P] array phenotype namesv (only if header line is specified in file),
    'vals'   : [N*P] array of phenotype-data,
    'iid'    : [N*2] array of family IDs and individual IDs
    --------------------------------------------------------------------------    
    '''
    data = SP.loadtxt(filename,dtype = 'str')
    if data[0,0] == 'FID':
        header = data[0,2::]
        data = data[1::]
    else:
        header = None
    iid = data[:,0:2]    
    data = data[:,2::]
    sel = data != missing
    sel = sel.reshape(-1)
    sel = SP.nonzero(sel)
    data = data[sel]
    iid = iid[sel,:]
    iid = iid.reshape(iid.shape[1], iid.shape[2])
    vals = SP.array(data,dtype = 'float')
    if pheno is not None:
        #TODO: sort and filter SNPs according to pheno.
        pass
    ret = {
            'header':header,
            'vals':vals,
            'iid':iid
            }
    return ret



def standardize(snps):
    N = snps.shape[0]
    S = snps.shape[1]
    for i in range(S):
        sel = snps[:,i] == snps[:,i]
        s = snps[sel,i]
        m = s.mean()
        sd = s.std()
        #print(m)
        #print(sd)
        snps[:,i] = (snps[:,i]-m)/sd
        snps[~sel,i] = 0
        #print(snps)
    return(snps)




def readBED(basefilename, blocksize = 1, start = 0, nSNPs = SP.inf, startpos = None, endpos = None, order  = 'F',standardizeSNPs=False,ipos = 2):
    '''
    read [basefilename].bed,[basefilename].bim,[basefilename].fam
    --------------------------------------------------------------------------
    Input:
    basefilename    : string of the basename of [basename].bed, [basename].bim,
                      and [basename].fam
    blocksize       : load blocksize SNPs at a time (default 1)
    start           : index of the first SNP to be loaded from the .bed-file
                      (default 0)
    nSNPs           : load nSNPs from the .bed file (default SP.inf, meaning all)
    startpos        : starting position of the loaded genomic region[chr,bpdist]
    endpos          : end-position of the loaded genomic region     [chr,bpdist]
    order           : memory layout of the returned SNP array (default 'F')
                      'F'   : Fortran-style column-major array (SNP-major)
                      'C'   : C-style row-major array (individual-major)
    standardizeSNPs : bool indeicator if the resulting SNP array is supposed to 
                      be zero-mean and unit-vatiance with mean imputed missing
                      values (default False)
    ipos            : the index of the position index to use (default 2)
                        1 : genomic distance
                        2 : base-pair distance
    --------------------------------------------------------------------------
    Output dictionary:
    'rs'     : [S] array rs-numbers
    'pos'    : [S*3] array of positions [chromosome, genetic dist, basepair dist]
    'snps'   : [N*S] array of snp-data
    'iid'    : [N*2] array of family IDs and individual IDs
    --------------------------------------------------------------------------    
    '''
    fam = basefilename+ '.fam'
    delimiter = deduce_delimiter(fam)
    fam = SP.loadtxt(fam,delimiter = delimiter,dtype = 'str',usecols=(0,1))
    bim = basefilename+'.bim'
    delimiter = deduce_delimiter(bim)
    bim = SP.loadtxt(bim,delimiter = delimiter,dtype = 'str',usecols = (0,1,2,3,4,5))
    rs = bim[:,1]
    pos = SP.array(bim[:,(0,2,3)],dtype = 'float')
    alleles = SP.array(bim[:,(4,5)],dtype='str')
    #pdb.set_trace()
    if startpos is not None:
        #pdb.set_trace()
        i_c = pos[:,0]==startpos[0]
        i_largerbp = pos[:,ipos]>=startpos[ipos]
        start = which(i_c * i_largerbp)
        while (start-1 >= 0 and pos[start-1,ipos] == startpos[ipos]):
            start = start -1
        i_c = pos[:,0]==endpos[0]
        i_smallerbp = pos[:,ipos]>=endpos[ipos]
        end = which(i_c * i_smallerbp)
        while (end+1 < pos.shape[0] and pos[end+1,ipos] == endpos[ipos]):
            end = end + 1
        nSNPs = end - start
        if (nSNPs<=0) or (end==0) or (start<=0):
            ret = {
                'pos':SP.zeros((0,3)),
                'rs':SP.zeros((0)),
                'iid':fam,
                'snps':SP.zeros((fam.shape[0],0))
                }
            return ret
        pass
    N = fam.shape[0]
    S = bim.shape[0]
    S_res = min(S,start + nSNPs)
    nSNPs = min(S-start,nSNPs)
    #if startpos is not None:
	#print("start: " + str(start))
	#print("end: " + str(end))
    #print("S_res: " + str(S_res))
    #print("nSNPs: " + str(nSNPs))
    if nSNPs<=0:
        ret = {
            'rs'     :rs[start:start],
            'pos'    :pos[start:start,:],
            #'snps'   :SNPs[0:N,start:start],
            'snps'   :SP.zeros((N,0)),
            'iid'    :fam
            }
        return ret
    SNPs = SP.zeros(((SP.ceil(0.25*N)*4),nSNPs),order=order)
    bed = basefilename + '.bed'
    with open(bed, "rb") as f:
        mode = f.read(2)
        if mode != 'l\x1b':
            raise Exception('No valid binary PED file')
        mode = f.read(1) #\x01 = SNP major \x00 = individual major
        if mode != '\x01':
            raise Exception('only SNP-major is implemented')
        startbit = SP.ceil(0.25*N)*start+3
        f.seek(startbit)
        for blockStart in SP.arange(0,nSNPs,blocksize):
            blockEnd = min(S,blockStart+blocksize)
            Sblock = min(nSNPs-blockStart,blocksize)
            nbyte = int(SP.ceil(0.25*N)*Sblock)
            bytes = SP.array(bytearray(f.read(nbyte))).reshape((SP.ceil(0.25*N),Sblock),order='F')
            
            SNPs[3::4,blockStart:blockEnd][bytes>=64]=SP.nan
            SNPs[3::4,blockStart:blockEnd][bytes>=128]=1
            SNPs[3::4,blockStart:blockEnd][bytes>=192]=2
            bytes=SP.mod(bytes,64)
            SNPs[2::4,blockStart:blockEnd][bytes>=16]=SP.nan
            SNPs[2::4,blockStart:blockEnd][bytes>=32]=1
            SNPs[2::4,blockStart:blockEnd][bytes>=48]=2
            bytes=SP.mod(bytes,16)
            SNPs[1::4,blockStart:blockEnd][bytes>=4]=SP.nan
            SNPs[1::4,blockStart:blockEnd][bytes>=8]=1
            SNPs[1::4,blockStart:blockEnd][bytes>=12]=2
            bytes=SP.mod(bytes,4)
            SNPs[0::4,blockStart:blockEnd][bytes>=1]=SP.nan
            SNPs[0::4,blockStart:blockEnd][bytes>=2]=1
            SNPs[0::4,blockStart:blockEnd][bytes>=3]=2
    
    if 0: #the binary format as described in the documentation (seems wrong)
        SNPs[3::4][bytes>=128]=SP.nan
        SNPs[3::4][bytes>=192]=1
        bytes=SP.mod(bytes,128)
        SNPs[3::4][bytes>=64]+=1
        bytes=SP.mod(bytes,64)
        SNPs[2::4][bytes>=32]=SP.nan
        SNPs[2::4][bytes>=48]=1
        bytes=SP.mod(bytes,32)
        SNPs[2::4][bytes>=16]+=1
        bytes=SP.mod(bytes,16)
        SNPs[1::4][bytes>=8]=SP.nan
        SNPs[1::4][bytes>=12]=1
        bytes=SP.mod(bytes,8)
        SNPs[1::4][bytes>=4]+=1
        bytes=SP.mod(bytes,4)
        SNPs[0::4][bytes>=2]=SP.nan
        SNPs[0::4][bytes>=3]=1
        bytes=SP.mod(bytes,2)
        SNPs[0::4][bytes>=1]+=1
    snps = SNPs[0:N,:]
    if standardizeSNPs:
        snps = standardize(snps)
    ret = {
            'rs'     :rs[start:S_res],
            'pos'    :pos[start:S_res,:],
            'alleles'    :alleles[start:S_res,:],
            'snps'   :snps,
            'iid'    :fam
            }
    return ret




def findIndex(ids, bedids): 
    N1 = ids.shape[0]
    com1 = SP.chararray(N1, itemsize=30)
    #com1 = {}
    N2 = bedids.shape[0]
    com2 = {}
    for i in range(N1):
        com1[i] = ids[i,0] + "_" + ids[i,1] 
        #com1[ ids[i,0] + "_" + ids[i,1] ] = i
    for i in range(N2):
        com2[ bedids[i,0] + "_" + bedids[i,1] ] = i
    if (N1 <= N2):
        index = SP.zeros(N1)
        count = 0
        for i in range(N1):
            try:
                ind = com2[ com1[i] ]
            except KeyError:
                continue
            index[count] = ind
            count = count + 1
    else:
        index = SP.zeros(N2)
        count = 0
        for i in range(N2):
            try:
                ind = com2[ com1[i] ]
            except KeyError:
                continue
            index[count] = ind
            count = count + 1
    index = index[0:count]
    #index = index[index != -1]
    return index.astype('I')
    
       


def filter(phe, bed):
    # for pheno
    index = findIndex(bed['iid'], phe['iid'])
    iid = phe['iid']
    iid = iid[index,:]
    vals = phe['vals']
    vals = vals[index]
    phe['iid'] = iid
    phe['vals'] = vals
    # for snp
    index = findIndex(phe['iid'], bed['iid'])
    iid = bed['iid']
    iid = iid[index,:]
    snps = bed['snps']
    snps = snps[index,:]
    bed['iid'] = iid
    bed['snps'] = snps
    return
     





#if __name__ == "__main__":
#    #datadir = "C:\\Users\\lippert\\Projects\\ARIC" ; basefilename = os.path.join(datadir, 'whiteMale')
#    datadir = "testdata" ; basefilename = os.path.join(datadir, 'test')
#    #bed = readBED(basefilename,blocksize = 1,nSNPs = 20000,start = 650000)
#    #bed = readBED(basefilename,blocksize = 1,nSNPs = 20000,start = 0)
#    #phe = loadPhen(basefilename + ".tab")
#    filter(phe, bed)
#    bed = readBED('Gaw14/all', startpos = [1,17.5599], endpos = [1,24.6047])
   


 
