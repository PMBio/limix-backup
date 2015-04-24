import numpy as np
from scipy.optimize import approx_fprime

_eps = np.sqrt(np.finfo(float).eps)

def mcheck_grad(func, grad, x0, epsilon=_eps,
                     allerrs=False):
    assert len(x0.shape) == 1
    errs = np.zeros(x0.size)
    x = x0.copy()
    for i in xrange(x0.size):

        def funci(x0i):
            x[i] = x0i
            r = func(x)
            x[i] = x0[i]
            return np.ravel(r)

        def gradi(x0i):
            x[i] = x0i
            g = np.ravel(grad(x))
            x[i] = x0[i]
            return g

        errs[i] = _vcheck_grad(funci, gradi, x0[i],
                                 epsilon)

    assert np.all(errs >= 0.0)
    if allerrs:
        return errs
    return np.max(errs)


def scheck_grad(theta_set, theta_get,
                value,
                grad, epsilon=_eps,
                allerrs=False):

    x0 = theta_get()
    assert type(x0) is np.ndarray

    def _func(x):
        theta_set(x)
        return value()

    def _grad(x):
        theta_set(x)
        return grad()

    return mcheck_grad(_func, _grad, x0, epsilon=_eps, allerrs=False)



########################### INTERNAL USE ONLY ###########################

def _check_grad(func, grad, x0, epsilon=_eps):
    return np.sqrt(sum((grad(x0) -
                   approx_fprime(x0, func, epsilon))**2))


def _vcheck_grad(func, grad, x0,
                   epsilon=_eps):

    assert len(func(x0).shape) == 1
    assert len(grad(x0).shape) == 1

    n = func(x0).size

    errs = np.zeros(n)

    for i in xrange(n):
        errs[i] = _check_grad(lambda x: func(x[0])[i], lambda x: grad(x[0])[i],
                             [x0], epsilon)


    return np.max(errs)
