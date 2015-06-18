import pdb
import scipy as sp
import scipy.linalg
import copy
import sys
import time
from limix.core.type.observed import Observed
from limix.core.type.cached import Cached, cached
from limix.core.covar import Covariance
from limix.core.mean import mean_base
from limix.core.covar.cov_reml import cov_reml
import limix.core.optimize.optimize_bfgs_new as OPT

import logging
logger = logging.getLogger(__name__)

class GP(Cached, Observed):
    """
    Gaussian Process regression class for linear mean (with REML)
    y ~ N(Wb,K)
    """

    def __init__(self, mean, covar):
        """
        covar:        Covariance function
        mean:         Linear Mean function
        """
        Cached.__init__(self)

        if not issubclass(type(mean), mean_base):
            raise TypeError('Parameter mean must have base_mean inheritance.')

        if not issubclass(type(covar), Covariance):
            raise TypeError('Parameter covar must have Covariance inheritance.')

        self.covar = covar
        self.mean  = mean
        self.Areml = cov_reml(self)
        self._observe()
        self.clear_all()

    def _observe(self):
        # different notification should be possible
        # e.g. for mean: pheno and designs
        # see GP2KronSum
        self.covar.register(self.clear_all)
        self.mean.register(self.clear_all)

    def clear_all(self):
        self.clear_Areml()
        self.clear_lml_terms()
        self.clear_lmlgrad_terms_i()
        self.clear_lmlgrad_terms()
        self.update_b()

    def clear_Areml(self):
        self._notify()

    def clear_lml_terms(self):
        self.clear_cache('KiW','yKiW','KiWb','Kiy','yKiy','yKiWb','LML')

    def clear_lmlgrad_terms_i(self):
        self.clear_cache('DiKKiy','DiKKiW','DiKKiWb',
                            'yKiy_grad_i','yKiWb_grad_i')

    def clear_lmlgrad_terms(self):
        self.clear_cache('yKiy_grad','yKiWb_grad','Areml_logdet_grad','LML_grad')

    def setParams(self,params):
        self.covar.setParams(params['covar'])

    def getParams(self):
        RV = {}
        RV['covar'] = self.covar.getParams()
        return RV

    ######################
    # Areml
    ######################
    def Areml_K(self):
        return sp.dot(self.mean.W.T,self.KiW())

    def Areml_K_grad_i(self,i):
        return -sp.dot(self.KiW().T,self.DiKKiW(i))

    #######################
    # LML terms
    #######################
    @cached
    def KiW(self):
        return self.covar.solve(self.mean.W)

    @cached
    def yKiW(self):
        return sp.dot(self.mean.y.T,self.KiW())

    # b is calculated here but cached in the mean?
    def update_b(self):
        if self.mean.n_covs > 0:
            self.mean.b = self.Areml.solve(self.yKiW().T)

    @cached
    def KiWb(self):
        return sp.dot(self.KiW(),self.mean.b)

    @cached
    def Kiy(self):
        return self.covar.solve(self.mean.y)

    @cached
    def yKiy(self):
        return (self.mean.y*self.Kiy()).sum()

    @cached
    def yKiWb(self):
        return (self.mean.y*self.KiWb()).sum()

    #######################
    # gradients
    #######################
    @cached
    def DiKKiy(self,i):
        return sp.dot(self.covar.K_grad_i(i),self.Kiy())

    @cached
    def DiKKiW(self,i):
        return sp.dot(self.covar.K_grad_i(i),self.KiW())

    @cached
    def DiKKiWb(self,i):
        return sp.dot(self.DiKKiW(i),self.mean.b)

    @cached
    def yKiy_grad_i(self,i):
        return -(self.Kiy()*self.DiKKiy(i)).sum()

    @cached
    def yKiWb_grad_i(self,i):
        rv = -2*(self.Kiy()*self.DiKKiWb(i)).sum()
        rv+= (self.KiWb()*self.DiKKiWb(i)).sum()
        return rv

    #######################
    # LML and gradients
    #######################

    @cached
    def LML(self):
        #const term to add?
        rv = 0.5*self.covar.logdet()
        rv += 0.5*self.yKiy()
        if self.mean.n_covs>0:
            rv += 0.5*self.Areml.logdet()
            rv -= 0.5*self.yKiWb()
        return rv

    @cached
    def yKiy_grad(self):
        n_params = self.getParams()['covar'].shape[0]
        RV = {'covar': sp.zeros(n_params)}
        for i in range(n_params):
            RV['covar'][i] = self.yKiy_grad_i(i)
        return RV

    @cached
    def yKiWb_grad(self):
        n_params = self.getParams()['covar'].shape[0]
        RV = {'covar': sp.zeros(n_params)}
        if self.mean.n_covs > 0:
            for i in range(n_params):
                RV['covar'][i] = self.yKiWb_grad_i(i)
        return RV

    @cached
    def Areml_logdet_grad(self):
        n_params = self.getParams()['covar'].shape[0]
        RV = {'covar': sp.zeros(n_params)}
        if self.mean.n_covs > 0:
            for i in range(n_params):
                RV['covar'][i] = self.Areml.logdet_grad_i(i)
        return RV

    @cached
    def LML_grad(self):
        n_params = self.getParams()['covar'].shape[0]
        RV = {'covar': sp.zeros(n_params)}
        for i in range(n_params):
            RV['covar'][i]  = 0.5*self.covar.logdet_grad_i(i)
            RV['covar'][i] += 0.5*self.yKiy_grad_i(i)
            if self.mean.n_covs > 0:
                RV['covar'][i] += 0.5*self.Areml.logdet_grad_i(i)
                RV['covar'][i] -= 0.5*self.yKiWb_grad_i(i)
        return RV

    def predict(self):
        R = None
        if self.covar.use_to_predict:
            Kcross = self.covar.Kcross()
            Kiyres = self.Kiy()-self.KiWb()
            R = sp.dot(Kcross,Kiyres)
        if self.mean.use_to_predict:
            _ = self.mean.predict()
            if R is None:
                R = _
            else:
                assert _.shape[0]==R.shape[0], 'Dimension mismatch'
                assert _.shape[1]==R.shape[1], 'Dimension mismatch'
                R += _
        return R

    #########################
    # OPTIMIZATION
    #########################
    def optimize(self,calc_ste=False,Ifilter=None,bounds=None,verbose=True,opts={},*args,**kw_args):
        #logger.info('Marginal likelihood optimization.')

        if verbose:
            print 'Marginal likelihood optimization.'
        t0 = time.time()
        conv, info = OPT.opt_hyper(self,Ifilter=Ifilter,bounds=bounds,opts=opts,*args,**kw_args)
        t1 = time.time()

        #if logger.levelno == logger.DEBUG:
        if verbose:
            #logger.debug('Time elapsed: %.2fs', t1-t0)
            print 'Converged:', conv
            print 'Time elapsed: %.2f s' % (t1-t0)
            grad = self.LML_grad()
            grad_norm = 0
            for key in grad.keys():
                grad_norm += (grad[key]**2).sum()
            grad_norm = sp.sqrt(grad_norm)
            print 'Log Marginal Likelihood: %.7f.' % self.LML()
            print 'Gradient norm: %.7f.' % grad_norm
            #logger.debug('Log Marginal Likelihood: %.7f.', self.LML())
            #logger.debug('Gradient norm: %.7f.', grad_norm)

        if calc_ste:
            #logger.info('Standard error calculation.')
            if verbose:
                print 'Standard errors calculation.'
            t0 = time.time()
            I_covar = self.covar.getFisherInf()
            I_mean = self.Areml.K()
            self.covar.setFIinv(sp.linalg.inv(I_covar))
            self.mean.setFIinv(sp.linalg.inv(I_mean))
            t1 = time.time()
            #logger.debug('Time elapsed: %.2fs', t1-t0)

        return conv, info

    def test_grad(self):
        from limix.utils.check_grad import mcheck_grad

        def func(x, i):
            params = self.getParams()
            params['covar'] = x
            self.setParams(params)
            return self.LML()

        def grad(x, i):
            params = self.getParams()
            params['covar'] = x
            self.setParams(params)
            grad = self.LML_grad()
            return grad['covar'][i]

        x0 = self.getParams()['covar']
        err = mcheck_grad(func, grad, x0)
        print err
        #np.testing.assert_almost_equal(err, 0., decimal=5)
