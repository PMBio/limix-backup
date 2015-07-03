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

import scipy as sp

# A simple function to calculat the minor allele frequency
# Here, we assume that M is either [0,1] or [0,1,2]
# The same principles applies if we have genotype likelihoods rather than hard thresholds

def calc_AF(M,major=0,minor=2):
    """calculate minor allelel frequency, by default assuming that minor==2"""
    if minor==2:
        Nhet   = (M==0).sum(axis=0)
        Nmajor = 2*(M==0).sum(axis=0)
        Nminor = 2*(M==2).sum(axis=0)
        af  = Nminor/sp.double(2*M.shape[0])
    else:
        Nmajor = (M==0).sum(axis=0)
        Nminor = (M==1).sum(axis=0)
        af  = Nminor/sp.double(1*M.shape[0])
    RV = {}
    RV['af'] = af
    RV['Nmajor'] = Nmajor
    RV['Nminor'] = Nminor
    return RV

def calc_LD(M,pos,i_start=[0],max_dist=1000000):
    """calculate linkage disequilibrium correlations:
    M: genotype matrix
    pos: position vector
    i_start: index to start from for LD calculation
    dist: distance
    """
    RV = []
    DIST = []
    for start in i_start:
        pos0 = pos[start]
        v0  = M[:,start]
        Iselect = sp.nonzero(sp.absolute(pos-pos0)<=max_dist)[0]
        rv = sp.zeros(len(Iselect))
        for i in xrange(len(Iselect)):
            rv[i] = (sp.corrcoef(v0,M[:,Iselect[i]])[0,1])**2
        #sort by distance
        dist = sp.absolute(pos[Iselect]-pos0)
        RV.extend(rv)
        DIST.extend(dist)
    RV = sp.array(RV)
    DIST = sp.array(DIST)
    II = DIST.argsort()
    return [DIST[II],RV[II]]