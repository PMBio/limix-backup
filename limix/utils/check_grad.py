import numpy as np
from scipy.optimize import approx_fprime

_eps = np.sqrt(np.finfo(float).eps)

def mcheck_grad(func, grad, x0, epsilon=_eps,
                     allerrs=False):
    assert len(x0.shape) == 1
    errs = np.zeros(x0.size)
    x = x0.copy()
    for i in range(x0.size):

        def funci(x0i):
            x[i] = x0i
            r = func(x, i)
            x[i] = x0[i]
            return np.ravel(r)

        def gradi(x0i):
            x[i] = x0i
            g = np.ravel(grad(x, i))
            x[i] = x0[i]
            return g

        errs[i] = _vcheck_grad(funci, gradi, x0[i],
                                 epsilon)

    assert np.all(errs >= 0.0)
    if allerrs:
        return errs
    return np.max(errs)


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

    for i in range(n):
        errs[i] = _check_grad(lambda x: func(x[0])[i], lambda x: grad(x[0])[i],
                             [x0], epsilon)


    return np.max(errs)
