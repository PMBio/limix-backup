import pdb
import scipy as sp
import scipy.linalg
import copy
import sys
import time
from gp_ls import GPLS
from limix.core.type.cached import cached
from limix.core.covar import Covariance
from limix.core.mean import MeanBase
from limix.core.covar.cov_reml import cov_reml

class GPMKS(GPLS):
    """
    Gaussian Process with a covariance and a mean that are sums of multiple kronecker products:
        vec(Y) ~ N( vec( \sum_i F_i B_i A_i), \sum_j C_j \kron R_j)
    Notation:
        N = number of samples
        P = number of traits
        Y = [N, P] phenotype matrix
        F_i = sample fixed effect design for term i
        A_i = trait fixed effect design for term i
        B_i = effect sizes of fixed effect term i
        C_j = column covariance matrix for term j
        R_j = column covariance matrix for noise term j 
    """

    def __init__(self, Y, C, R, F=None, A=None):
        """
        Args:
            Y:      [N, P] phenotype matrix
            C:      list of limix trait covariances. 
                    Each term must be a limix covariance with dimension P 
            C:      list of row covariance matrices. 
                    Each term must be a numpy covariane matrix with dimension P 
            F:      list of sample fixed effect designs.
                    Each term must have first dimension N
            A:      list of trait fixed effect design.
                    Each term must have second dimension P
        """
        # assert types
        assert_type(Y, NP.ndarray, 'Y')
        assert len(C)==len(R), 'Dimension mismatch'
        for i in range(len(C)):
            assert_subtype(C[i], Covariance, 'C[%d]' % i)
            assert_type(R[i], NP.ndarray, 'R[%d]' % i)

        assert F is None and A is None, 'fixed effects not supported yet'

        # build covariance matrix
        #Iok = vec(~sp.isnan(Y))[:,0]
        #if veIok.all():     Iok = None
        covar = SumCov(*[KronCov(C[i], R[i]) for i in range(len(C))])

        # init GPLS
        GP.__init__(self, Y, covar)

    def _observe(self):
        #TODO: check if we need row and col covariances
        self.covar.register(self.row_cov_has_changed, 'row_cov')
        self.covar.register(self.col_cov_has_changed, 'col_cov')
        self.mean.register(self.pheno_has_changed, 'pheno')
        self.mean.register(self.designs_have_changed, 'designs')

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
