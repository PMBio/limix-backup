import sys
sys.path.insert(0,'./../../..')

from limix.core.mean.mean_base import mean_base as lin_mean
from limix.core.covar.sqexp import sqexp
from limix.core.gp.gp_base_new import gp

import ipdb
import scipy as sp
import scipy.linalg as LA 
import time as TIME
import copy
import pylab as pl
pl.ion()


if __name__ == "__main__":

    # generate data
    N = 400
    s_x = 0.05
    s_y = 0.1
    X = (sp.linspace(0,2,N)+s_x*sp.randn(N))[:,sp.newaxis]
    Y = sp.sin(X)+s_y*sp.randn(N,1)
    #pl.plot(x,y,'x')

    ipdb.set_trace()

    # define mean term
    F = 1.*(sp.rand(N,2)<0.2)
    mu = lin_mean(Y,F)

    # define covariance matrices
    covar = sqexp(X)
    
    ipdb.set_trace()

    if 1:
        # compare with mtSet implementation
        params = {}
        params['Cg']   = sp.randn(int(0.5*P*(P+1)))
        params['Cn']   = sp.randn(int(0.5*P*(P+1)))

        print "check gradient with gp2kronSum"
        gp = gp2kronSum(mu,Cg,Cn,XX)
        gp.setParams(params)

        if 0:
            gp.set_reml(False)

        print "test optimization"
        start = TIME.time()
        conv,info = OPT.opt_hyper(gp,params,factr=1e3)
        print 'Reml GP:', TIME.time()-start
        
        if mtSet_present:
            params1 = copy.copy(params)
            params1['mean'] = sp.zeros(mu1.getParams().shape[0])
            gp1 = gp2kronSumMtSet(mu1,Cg1,Cn1,XX)
            gp1.setParams(params1)
            start = TIME.time()
            conv1,info = OPT.opt_hyper(gp1,params1,factr=1e3)
            print 'Old GP:', TIME.time()-start

        print conv

        ipdb.set_trace()

    if 1:
        # no fixed 
        mu = mean(Y)
        params = {}
        params['Cg']   = sp.randn(int(0.5*P*(P+1)))
        params['Cn']   = sp.randn(int(0.5*P*(P+1)))

        ipdb.set_trace()
        gp = gp2kronSum(mu,Cg,Cn,XX)
        gp.setParams(params)
        gp.LML()
        gp.LMLgrad()

