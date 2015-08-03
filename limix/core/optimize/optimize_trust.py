from scipy.optimize import fmin_l_bfgs_b as optimize
import pdb
import scipy as SP
import scipy.linalg as LA
import logging as LG
import numpy as np
import time

def param_dict_to_list(dict,skeys=None):
    """convert from param dictionary to list"""
    #sort keys
    RV = SP.concatenate([dict[key].flatten() for key in skeys])
    return RV
    pass

def param_list_to_dict(list,param_struct,skeys):
    """convert from param dictionary to list
    param_struct: structure of parameter array
    """
    RV = []
    i0= 0
    for key in skeys:
        val = param_struct[key]
        shape = SP.array(val)
        np = shape.prod()
        i1 = i0+np
        params = list[i0:i1].reshape(shape)
        RV.append((key,params))
        i0 = i1
    return dict(RV)

def checkgrad(f, fprime, x, *args,**kw_args):
    """
    Analytical gradient calculation using a 3-point method

    """
    LG.debug("Checking gradient ...")
    import numpy as np

    # using machine precision to choose h
    eps = np.finfo(float).eps
    step = np.sqrt(eps)*(x.min())

    # shake things up a bit by taking random steps for each x dimension
    h = step*np.sign(np.random.uniform(-1, 1, x.size))

    f_ph = f(x+h, *args, **kw_args)
    f_mh = f(x-h, *args, **kw_args)
    numerical_gradient = (f_ph - f_mh)/(2*h)
    analytical_gradient = fprime(x, *args, **kw_args)
    ratio = (f_ph - f_mh)/(2*np.dot(h, analytical_gradient))

    h = np.zeros_like(x)
    for i in range(len(x)):
        pdb.set_trace()
	h[i] = step
	f_ph = f(x+h, *args, **kw_args)
	f_mh = f(x-h, *args, **kw_args)
	numerical_gradient = (f_ph - f_mh)/(2*step)
	analytical_gradient = fprime(x, *args, **kw_args)[i]
	ratio = (f_ph - f_mh)/(2*step*analytical_gradient)
	h[i] = 0
        LG.debug("[%d] numerical: %f, analytical: %f, ratio: %f" % (i, numerical_gradient,analytical_gradient,ratio))



def opt_hyper(gpr,theta=1e-2,max_iter=None,alpha=1,tr=None,returnLML=False,noH=False,debug=False):

    #maxiter
    if max_iter==None:
        max_iter = 100 * gpr.getParams()['covar'].shape[0] 

    LML = SP.zeros(max_iter)
    if tr is not None: 
        max_ap = tr * SP.ones(gpr.getParams()['covar'].shape[0])
        min_ap = - tr * SP.ones(gpr.getParams()['covar'].shape[0])

    for i in range(max_iter):
        if debug:
            print 'iteration', i
            print 'params:', gpr.getParams()['covar']
        grad = gpr.LML_grad()['covar']

        if returnLML:
            LML[i] = gpr.LML()
            print 'LML:', gpr.LML()

        if debug:
            print 'LMLgrad', (grad**2).mean()

        conv = SP.absolute(grad).max() < theta
        if conv:    break

        # update params 
        if noH:
            p = - grad
        else:
            H = gpr.AIM()
            p = SP.dot(LA.pinv(H), grad)
            #try:
            #    H = gpr.AIM()
            #    L = LA.cholesky(-H)
            #    p = - alpha * LA.cho_solve((L, False), grad)
            #except:
            #    pdb.set_trace()
        ap = alpha * p
        if tr is not None:
            ap = SP.clip(ap, min_ap, max_ap)
        params = {'covar': gpr.getParams()['covar'] + ap}
        gpr.setParams(params)

    RV = {'n_iter': i, 'grad': grad}

    if returnLML:
        RV['LML'] = LML[:i+1]

    return conv, RV 

