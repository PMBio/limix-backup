import pdb
import scipy as sp
import scipy.linalg
import copy
import sys
import time
from limix.core.type.observed import Observed
from limix.core.type.cached import Cached, cached
from limix.core.covar import Covariance
from limix.core.mean import MeanBase
from limix.core.covar.cov_reml import cov_reml
import limix.core.optimize.optimize_trust as OPT

import logging
logger = logging.getLogger(__name__)


class GPLS(Cached, Observed):
    """
    Gaussian Process regression class for linear mean (with REML) where all computation is based on solving linear systems
    y ~ N(Wb,K)
    Notation:
        N = number of samples
        K = number of fixed effects
        y = phenotype vector
        W = fixed effect design
        K = covariance function
    """

    def __init__(self, y, covar):
        """
        Args:
            Y:          [N, P] phenotype matrix
            covar:      Limix covariance function
        """
        Cached.__init__(self)

        if not issubclass(type(covar), Covariance):
            raise TypeError('Parameter covar must have Covariance '
                            'inheritance.')

        self.covar = covar
        self.mean = MeanBase(y) 
        self.Areml = cov_reml(self)
        self._observe()

    def _observe(self):
        # different notification should be possible
        # e.g. for mean: pheno and designs
        # see GP2KronSum
        self.covar.register(self.clear_all)
        self.mean.register(self.clear_all)

    def clear_all(self):
        self._notify() # notify Areml
        self.clear_cache('gp_base')
        self.update_b()

    def setParams(self, params):
        self.covar.setParams(params['covar'])

    def getParams(self):
        RV = {}
        RV['covar'] = self.covar.getParams()
        return RV

    ######################
    # Areml
    ######################
    def Areml_K(self):
        return sp.dot(self.mean.W.T, self.KiW())

    def Areml_K_grad_i(self, i):
        return -sp.dot(self.KiW().T, self.DiKKiW(i))

    #######################
    # LML terms
    #######################
    def update_b(self):
        if self.mean.n_covs > 0:
            self.mean.b = self.Areml.solve(self.yKiW().T)

    @cached('gp_base')
    def Kiy(self):
        return self.covar.solve_ls(self.mean.y)

    @cached('gp_base')
    def yKiy(self):
        return (self.mean.y*self.Kiy()).sum()

    #@cached('gp_base')
    #def KiW(self):
    #    return self.covar.solve(self.mean.W)

    #@cached('gp_base')
    #def yKiW(self):
    #    return sp.dot(self.mean.y.T, self.KiW())

    #@cached('gp_base')
    #def KiWb(self):
    #    return sp.dot(self.KiW(), self.mean.b)

    #@cached('gp_base')
    #def yKiWb(self):
    #    return (self.mean.y*self.KiWb()).sum()

    #######################
    # gradients
    #######################
    #@cached('gp_base')
    #def DiKKiy(self, i):
    #    return sp.dot(self.covar.K_grad_i(i), self.Kiy())

    @cached('gp_base')
    def DKKiy(self):
        n_params = self.getParams()['covar'].shape[0]
        R = sp.zeros((self.mean.y.shape[0], n_params))
        for i in range(n_params):
            R[:, i:i+1] = sp.dot(self.covar.K_grad_i(i), self.Kiy()) 
        return R

    @cached('gp_base')
    def DDKKiy(self):
        n_params = self.getParams()['covar'].shape[0]
        R = sp.zeros((self.mean.y.shape[0], n_params, n_params))
        for i in range(n_params):
            R[:, i, i] = sp.dot(self.covar.K_hess_i_j(i, i), self.Kiy()[:,0])
            for j in range(i):
                R[:, i, j] = sp.dot(self.covar.K_hess_i_j(i, j), self.Kiy()[:,0])
                R[:, j, i] = R[:, i, j] 
        return R

    #@cached('gp_base')
    #def yKiy_grad_i(self, i):
    #    return -(self.Kiy()*self.DiKKiy(i)).sum()

    #@cached('gp_base')
    #def DiKKiW(self, i):
    #    return sp.dot(self.covar.K_grad_i(i), self.KiW())
    #    pass

    #@cached('gp_base')
    #def DiKKiWb(self, i):
    #    return sp.dot(self.DiKKiW(i), self.mean.b)
 
    #@cached('gp_base')
    #def yKiWb_grad_i(self, i):
    #    rv = -2*(self.Kiy()*self.DiKKiWb(i)).sum()
    #    rv += (self.KiWb()*self.DiKKiWb(i)).sum()
    #    return rv

    #######################
    # LML and gradients
    #######################

    @cached('gp_base')
    def LML(self):
        # const term to add?
        rv = 0.5*self.covar.logdet()
        rv += 0.5*self.yKiy()
        #if self.mean.n_covs > 0:
        #    rv += 0.5*self.Areml.logdet()
        #    rv -= 0.5*self.yKiWb()
        return rv

    @cached('gp_base')
    def yKiy_grad(self):
        return -(self.Kiy() * self.DKKiy()).sum(0)

    #@cached('gp_base')
    #def yKiWb_grad(self):
    #    n_params = self.getParams()['covar'].shape[0]
    #    RV = {'covar': sp.zeros(n_params)}
    #    if self.mean.n_covs > 0:
    #        for i in range(n_params):
    #            RV['covar'][i] = self.yKiWb_grad_i(i)
    #    return RV

    @cached('gp_base')
    def Areml_logdet_grad(self):
        n_params = self.getParams()['covar'].shape[0]
        RV = {'covar': sp.zeros(n_params)}
        if self.mean.n_covs > 0:
            for i in range(n_params):
                RV['covar'][i] = self.Areml.logdet_grad_i(i)
        return RV

    @cached('gp_base')
    def LML_grad(self):
        RV  = 0.5 * self.covar.sample_logdet_grad()
        RV += 0.5 * self.yKiy_grad()
        RV = {'covar': RV} 
        #if self.mean.n_covs > 0:
        #    RV['covar'][i] += 0.5*self.Areml.logdet_grad_i(i)
        #    RV['covar'][i] -= 0.5*self.yKiWb_grad_i(i)
        return RV

    @cached('gp_base')
    def AIM(self):
        KiDKKiy = self.covar.solve_ls(self.DKKiy())
        R = - 0.25 * self.covar.sample_trKiDDK()
        R-= 0.5 * sp.dot(self.DKKiy().T, KiDKKiy)
        R+= 0.25 * sp.tensordot(self.Kiy(), self.DDKKiy(), axes=(0,0))[0]
        return R

    def predict(self):
        R = None
        if self.covar.use_to_predict:
            Kcross = self.covar.Kcross()
            Kiyres = self.Kiy()-self.KiWb()
            R = sp.dot(Kcross, Kiyres)
        if self.mean.use_to_predict:
            _ = self.mean.predict()
            if R is None:
                R = _
            else:
                assert _.shape[0] == R.shape[0], 'Dimension mismatch'
                assert _.shape[1] == R.shape[1], 'Dimension mismatch'
                R += _
        return R

    #########################
    # OPTIMIZATION
    #########################
    def optimize(self, calc_ste=False, verbose=True, *args, **kw_args):
        # logger.info('Marginal likelihood optimization.')

        if verbose:
            print 'Marginal likelihood optimization.'
        t0 = time.time()
        conv, info = OPT.opt_hyper(self, *args, **kw_args)
        t1 = time.time()

        # if logger.levelno == logger.DEBUG:
        if verbose:
            # logger.debug('Time elapsed: %.2fs', t1-t0)
            print 'Converged:', conv
            print 'n_iter:', info['n_iter']
            print 'Time elapsed: %.2f s' % (t1-t0)
            grad = self.LML_grad()
            grad_norm = 0
            for key in grad.keys():
                grad_norm += (grad[key]**2).sum()
            grad_norm = sp.sqrt(grad_norm)
            print 'Log Marginal Likelihood: %.7f.' % self.LML()
            print 'Gradient norm: %.7f.' % grad_norm
            # logger.debug('Log Marginal Likelihood: %.7f.', self.LML())
            # logger.debug('Gradient norm: %.7f.', grad_norm)

        if calc_ste:
            # logger.info('Standard error calculation.')
            if verbose:
                print 'Standard errors calculation.'
            t0 = time.time()
            I_covar = self.covar.getFisherInf()
            I_mean = self.Areml.K()
            self.covar.setFIinv(sp.linalg.pinv(I_covar))
            self.mean.setFIinv(sp.linalg.pinv(I_mean))
            t1 = time.time()
            # logger.debug('Time elapsed: %.2fs', t1-t0)

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
        # np.testing.assert_almost_equal(err, 0., decimal=5)
