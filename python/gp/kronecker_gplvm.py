"""
GPLVM module with kronecker covaraince structrure
Code assumes a data matrix Y such that
vec Y ~ N(0,Kc \otimes Kr + sigma^2 \eye)

The implementation frequenly assumes that we use Y.ravel rather than vec Y which are transposed, i..e

Y.ravel ~ N(o, Kr \otimes Kc + sigma^2 eye)
See also ./../derivations/derivations.tex
"""





import sys
#add path for new pygp
sys.path.insert(0,'./../../pygp/')
sys.path.append('./..')

#gplvm model
from pygp.covar import fixed,linear, noise, combinators
from pygp.gp import gplvm
import pygp.gp.gplvm as GPLVM
import pygp.optimize as OPT
#import optimize_test as OPT2
#customized covariance function:
import pygp.priors.lnpriors as lnpriors
import pygp.likelihood as lik
#import linear_regression as linreg
import logging as LG
import string, pickle
import scipy as SP
import numpy as N
import copy
import scipy.stats as st
import pdb
import copy
from pygp.linalg import *
jitter =1E-6
import numpy.random as random


VERBOSE=False

def getK_verbose(KV,noise=True):
    if 'K' not in KV.keys():
        K = SP.kron(KV['Kr'],KV['Kc'])
        if noise:
            K += SP.diag(KV['Knoise'])
        Ki = SP.linalg.inv(K)
        KV['K'] = K
        KV['Ki'] =Ki
    return [KV['K'],KV['Ki']]

def check_dist(A,B):
    assert SP.absolute((A-B)/A).max()<1E-4, 'outch'

def krondiag(v1,v2):
    """calcualte diagonal of kronecker(diag(v1),diag(v2)))
    note that this returns a non-flattened matrix
    """
    M1 = SP.tile(v1[:,SP.newaxis],[1,v2.shape[0]])
    M2 = SP.tile(v2[SP.newaxis,:],[v1.shape[0],1])
    M1 *= M2
    #RV  = (M1).ravel()
    #naive:
    #r=SP.kron(SP.diag(v1), SP.diag(v2)).diagonal()
    #pdb.set_trace()
    return M1

def kronvec(A,B,X):
    """calculate vec (A x B) vec X"""
    #what we are really doing Y ;-)
    #rv_ = SP.dot(SP.kron(A,B),vec(X))
    rv = vec(SP.dot(SP.dot(B,X),A.T))
    return rv

def kronravel(A,B,X):
    """calculate  ((A x B) X.ravel()).reshape()
    These are all the same:

    SP.dot(SP.kron(k2,k1),x.flatten()).reshape(x.shape)
    Ivec(SP.dot(SP.kron(k1,k2),vec(x)),x.shape)
    SP.dot(k2,SP.dot(x,k1.T))


    These as well:
    SP.dot(SP.kron(k2,k1),x.flatten())
    SP.dot(k2,SP.dot(x,k1.T)).flatten()

    And these:
    SP.dot(SP.kron(k1,k2),vec(x))
    vec(SP.dot(k2,SP.dot(x,k1.T)))
    """
    #what we are really doing Y ;-)
    #rv_ = SP.reshape(SP.dot(SP.kron(A,B),X.flatten()))
    rv = SP.dot(SP.dot(A,X),B.T)
    return rv


class KroneckerGPLVM(GPLVM.GPLVM):
    """GPLVM for kronecker type covariance structures
    This class assumes the following model structure
    \vecY ~ GP(0, Kx \otimes Kd + \sigma^2 \unit)
    The standard covariance function inherited form the gplvm class is used to for Kx; covar_func_D specifices Kd
    """
    pass

    def __init__(self,covar_func=None,covar_func_r=None,covar_func_c=None,likelihood=None,**kw_args):
        """initialize gplvm object, covar_func/covar_func_r: row covarince, covar_func_c: column covariance

        Y ~ N(0, covar_fun_c \kronecker covar_func_r)
        Where covar_func_c relates to the columns of Y and convar_func_rw to the rows of the data matrix
        """
        if ~(likelihood is None):
            assert isinstance(likelihood,lik.GaussLikISO), 'only Gaussian likleihood models are supported'
        
        #call constructor of GLM
        if covar_func is not None:
            covar_func_r  = covar_func
        if covar_func_r is not None:
            covar_func = covar_func_r
        super(GPLVM.GPLVM, self).__init__(covar_func=covar_func_r,likelihood=likelihood,**kw_args)
        self.covar_r = covar_func_r
        self.covar_c = covar_func_c
        
    def setData(self,x_c=None,x_r=None,x=None,gplvm_dimensions_r=None,gplvm_dimensions_c=None,y=None,**kw_args):
        #previous interfaces used x,y; now e add x_r/x_c assuming x_r=x
        self.y=y
        self.n=y.shape[0]
        self.d=y.shape[1]
        if x_r is not None:
            x = x_r
        else:
            x_r = SP.zeros([self.n,0])
        if x is not None:
            x_r = x
        #GPLVM.GPLVM.setData(self,x = x_r,**kw_args)
        #inputs for second covariance if applicable
        if x_c is not None:
            assert x_c.shape[0]==self.d, 'dimension missmatch'
        else:
            x_c = SP.zeros([self.d,0])
        self.x_c = x_c
        self.x_r = x_r
        #useful input matrix which hold the size of the entire kronecker structure
        self.xx = SP.zeros([self.x_r.shape[0]*self.x_c.shape[0],0])
        #store rehsaped view of Y
        self.nd = self.n*self.d
        if gplvm_dimensions_r is None:
            gplvm_dimensions_r = SP.arange(self.x_r.shape[1])
        if gplvm_dimensions_c is None:
            gplvm_dimensions_c = SP.arange(self.x_c.shape[1])
        #store dimensions
        self.gplvm_dimensions_r = gplvm_dimensions_r
        self.gplvm_dimensions_c = gplvm_dimensions_c
        self._invalidate_cache()

    def _update_inputs(self, hyperparams):
        """update the inputs from gplvm models if supplied as hyperparms"""
        if 'x_r' in hyperparams.keys():
            self.x_r[:, self.gplvm_dimensions_r] = hyperparams['x_r']
        if 'x_c' in hyperparams.keys():
            self.x_c[:, self.gplvm_dimensions_c] = hyperparams['x_c']
        pass
    

    def get_covariances(self, hyperparams):
        """get covariance structures and do necessary computations"""
        #check individaul components of hyperparams to avoid redunand recomputations
        #if full covariance parameter set is identical do nothing
        if self._is_cached(hyperparams) and not self._active_set_indices_changed:
            pass
        else:
            if self._covar_cache is None:
                self._covar_cache = {}
            CC = self._covar_cache
            #else partial update where needed
            #row covaraince update?
            if not self._is_cached(hyperparams,['covar_r']):
                Kr = self.covar_r.K(hyperparams['covar_r'],self.x_r)
                [Sr,Ur] = SP.linalg.eigh(Kr)
                CC['Kr'] = Kr
                CC['Sr'] = Sr
                CC['Ur'] = Ur
            if not self._is_cached(hyperparams,['covar_c']):
                Kc = self.covar_c.K(hyperparams['covar_c'],self.x_c)
                [Sc,Uc] = SP.linalg.eigh(Kc)
                CC['Kc'] = Kc
                CC['Sc'] = Sc
                CC['Uc'] = Uc
            #everything else is cheap and update always
             
            #0. evaluate likelihood
            Knoise = self.likelihood.Kdiag(hyperparams['lik'],self.xx)
                        
            # K1  = u1 * SP.diag(s1) * u1.T
            # K1_ = SP.dot(u1,SP.dot(SP.diag(s1),u1.T))
            #3. calculate rotated data matrix
            y_rot  = kronravel(CC['Ur'].T,CC['Uc'].T,self.y)
            #4. get Si which we need frequently
            Si = 1./(krondiag(CC['Sr'],CC['Sc']) + Knoise[0])
            YSi = y_rot*Si
            #4. store everything
            CC['Knoise'] =Knoise
            CC['Si']     =Si
            CC['y_rot']  =y_rot
            CC['YSi']    =YSi
            CC['hyperparams'] = copy.deepcopy(hyperparams)
            if VERBOSE:
                print "costly verbose debugging on"
                #check corectness of rotation
                y_rot_vec  = (y_rot).ravel()
                y_rot_vec2 = SP.dot(self.y.ravel(),SP.kron(Ur,Uc))
                check_dist(y_rot_vec,y_rot_vec2)
                
                #3. precalc rotated data matrix for efficient lml evaluation
                K = SP.kron(Kr,Kc)
                [S,U] = SP.linalg.eigh(K)
                #SS kron
                SS = SP.kron(Sr,Sc)
                UU = SP.kron(Ur,Uc)
                iSS = SS.argsort()
                SS = SS[iSS]
                UU = UU[:,iSS]
                #asserts decomposition
                assert SP.absolute(K-SP.dot(UU,SP.dot(SP.diag(SS),UU.T))).max()<1E-4, 'no valid SVD'
                assert SP.absolute(SP.dot(UU,UU.T)-SP.eye(K.shape[0])).max()<1E-4
                assert SP.absolute(SP.dot(UU.T,UU)-SP.eye(K.shape[0])).max()<1E-4
                self._covar_cache['U'] = U
                self._covar_cache['S'] = S
        return self._covar_cache



    def predict(self,hyperparams,xstar_r=None,xstar_c=None,var=False,hyperparams_star=None):
        """
        Predict mean and variance for given **Parameters:**

        hyperparams : {}
            hyperparameters in logSpace

        xstar    : [double]
            prediction inputs

        var      : boolean
            return predicted variance
        
        interval_indices : [ int || bool ]
            Either scipy array-like of boolean indicators, 
            or scipy array-like of integer indices, denoting 
            which x indices to predict from data.

        hyperparams_star: optional alternative hyperparameters for cross covariance
        
        output   : output dimension for prediction (0)
        """
        
        if var:
            print "predictive varinace not supported yet"

        if hyperparams_star is None:
            hyperparams_star = hyperparams
        
        #1. get covariance sturcture
        KV = self.get_covariances(hyperparams)

        #cross covariance:
        Kr_star = self.covar_r.K(hyperparams_star['covar_r'],self.x_r,xstar_r)
        Kc_star = self.covar_c.K(hyperparams_star['covar_c'],self.x_c,xstar_c)
        
        Kr_starU = SP.dot(Kr_star,KV['Ur'])
        Kc_starU = SP.dot(Kc_star,KV['Uc'])
        mu = kronravel(Kr_starU,Kc_starU,KV['YSi'])

        if VERBOSE:
            #trivial computations
            [K,Ki] = getK_verbose(KV)
            Kstar = SP.kron(Kr_star,Kc_star)
            yt = SP.dot(Ki,self.y.ravel())
            mu_slow = SP.dot(Kstar,yt)
            check_dist(mu.ravel(),mu_slow)
        return mu
        
    #overwrite calculation of lml for covariance parameters
    def _LML_covar(self, hyperparams):
        #calculate marginal likelihood of kronecker GP

        #1. get covariance structures needed:
        try:
            KV = self.get_covariances(hyperparams)
        except linalg.LinAlgError:
            LG.error("exception caught (%s)" % (str(hyperparams)))
            return 1E6
        #2. build lml 
        LML = 0
        LMLc = 0.5* self.nd * SP.log(2.0 * SP.pi)
        #constant part of negative lml
        #quadratic form
        Si = KV['Si']
       
        LMLq = 0.5 * SP.dot(KV['y_rot'].ravel(),KV['YSi'].ravel() )
        #determinant stuff
        LMLd = -0.5 * SP.log(Si).sum()

        if VERBOSE:
            print "costly verbose debugging on"
            K = SP.kron(KV['Kr'],KV['Kc']) + SP.diag(KV['Knoise'])
            Ki = SP.linalg.inv(K)
            LMLq_ = 0.5* SP.dot(SP.dot(self.y.ravel(),Ki),self.y.ravel())
            LMLd_ = 0.5* 2 * SP.log(SP.linalg.cholesky(K).diagonal()).sum()
            check_dist(LMLq,LMLq_)
            check_dist(LMLd,LMLd_)
            

        return LMLc+LMLq+LMLd
    

    def _LMLgrad_covar(self, hyperparams):
        #1. get inggredients for computations
        try:   
            KV = self.get_covariances(hyperparams)
        except linalg.LinAlgError:
            LG.error("exception caught (%s)" % (str(hyperparams)))
            return {'covar_r':SP.zeros(len(hyperparams['covar_r'])),'covar_c':SP.zeros(len(hyperparams['covar_c']))}

        if VERBOSE:
            [K,Ki] = getK_verbose(KV)
        #gradients with respect to row covaraince
        RV = {}

        #row:
        logtheta_r = hyperparams['covar_r']
        LMLgrad_r = SP.zeros(len(logtheta_r))
        for i in xrange(len(logtheta_r)):
            #get derivative matrix with respect to hyperparam i:
            Kd = self.covar_r.Kgrad_theta(hyperparams['covar_r'], self.x_r, i)           
            #calc
            grad_logdet= 0.5*self._gradLogDet(hyperparams,Kd,columns =False )
            grad_quad = 0.5*self._gradQuadrForm(hyperparams,Kd,columns =False )
            LMLgrad_r[i] = grad_logdet + grad_quad
            
            if VERBOSE:
                print "expensive gradcheck"
                #1. logdet term
                dKl = SP.kron(Kd,KV['Kc'])
                grad_logdet_ = 0.5 * SP.dot(Ki,dKl).trace()
                check_dist(grad_logdet,grad_logdet_)
                #2. quadratic part
                dKq = SP.dot(SP.dot(Ki,dKl),Ki)
                grad_quad_ = - 0.5* SP.dot(SP.dot(self.y.ravel(),dKq),self.y.ravel())
                check_dist(grad_quad,grad_quad_)    

        RV['covar_r'] = LMLgrad_r

        #column:
        logtheta_c = hyperparams['covar_c']
        LMLgrad_c = SP.zeros(len(logtheta_c))
        for i in xrange(len(logtheta_c)):
            #get derivative matrix with respect to hyperparam i:
            Kd = self.covar_c.Kgrad_theta(hyperparams['covar_c'], self.x_c, i)
            #calc
            grad_logdet= 0.5*self._gradLogDet(hyperparams,Kd,columns =True )
            grad_quad = 0.5*self._gradQuadrForm(hyperparams,Kd,columns =True )
            LMLgrad_c[i] = grad_logdet + grad_quad

            if VERBOSE:
                print "expensive gradcheck"
                #1. logdet term
                dKl = SP.kron(KV['Kr'],Kd)
                grad_logdet_ = 0.5 * SP.dot(Ki,dKl).trace()
                grad_logdet= 0.5*self._gradLogDet(hyperparams,Kd,columns =True )
                check_dist(grad_logdet,grad_logdet_)
                #2. quadratic part
                dKq = SP.dot(SP.dot(Ki,dKl),Ki)
                grad_quad_ = - 0.5* SP.dot(SP.dot(self.y.ravel(),dKq),self.y.ravel())
                grad_quad=0.5*self._gradQuadrForm(hyperparams,Kd,columns =True )
                check_dist(grad_quad,grad_quad_)

        RV['covar_c'] = LMLgrad_c
        return RV


    def _LMLgrad_lik(self,hyperparams):
        """derivative of the likelihood parameters"""
        try:   
            KV = self.get_covariances(hyperparams)
        except linalg.LinAlgError:
            LG.error("exception caught (%s)" % (str(hyperparams)))
            return {'lik':SP.zeros(len(hyperparams['lik']))}

        #note that we implicitly assume that Knoise = sigma^2* eye(n)
        logtheta = hyperparams['lik']
        LMLgrad = SP.zeros(1)
        dK=SP.exp(2.*logtheta[0])*2. #this is a constant
        Si=KV['Si']
        grad_logdet = 0.5*dK*Si.sum()
        YSi = KV['YSi'].ravel()
        grad_quad=-0.5*SP.dot(YSi,YSi)*dK
        LMLgrad[0] = grad_logdet + grad_quad

        if VERBOSE:
            print "expensive gradient lik"
            [K,Ki] = getK_verbose(KV)
            logtheta = hyperparams['lik']
            LMLgrad_ = SP.zeros(len(logtheta))
            Kd = self.likelihood.Kgrad_theta(logtheta, self.xx, 0)
            #1. logdet term
            grad_logdet_ = 0.5 * SP.dot(Ki,Kd).trace()
            #2. quadratic part
            dKq = SP.dot(SP.dot(Ki,Kd),Ki)
            grad_quad_ = - 0.5* SP.dot(SP.dot(self.y.ravel(),dKq),self.y.ravel())
            LMLgrad_[0] = grad_logdet_ + grad_quad_
            check_dist(LMLgrad[0],LMLgrad_[0])            
            
        RV = {'lik': LMLgrad}
        return RV

            
        
    
    def _LMLgrad_x(self, hyperparams):
        """grandient with respect to covarince inputs"""
        if not (('x_r' in hyperparams) or ('x_c' in hyperparams)):
            return {}
        try:
            KV = self.get_covariances(hyperparams)
        except linalg.LinAlgError:
            LG.error("exception caught (%s)" % (str(hyperparams)))
            return {'x_r': SP.zeros(hyperparams['x_r'].shape)}
        RV={}

        #1. GPLVM for row covariance?
        if 'x_r' in hyperparams:
            dlMl = SP.zeros_like(self.x_r)
                        #cycle through GPLVM data dimensions
            for i in xrange(len(self.gplvm_dimensions_r)):
                #note, not all dimensions may be learnt, hence gplvm_dimeensions index
                d = self.gplvm_dimensions_r[i]
                #K(X,X') , derivarive wr.t. first argument for dimension d
                dKx = self.covar_r.Kgrad_x(hyperparams['covar_r'], self.x_r, self.x_r, d)
                #print ('check_dist in _LML_grad_x')
                #check_dist(dKx[0,:],dKx[1,:])
                #pdb.set_trace()
                dgradLogdetX = self._gradLogDetX(hyperparams,dKx,columns=False)
                dgradQuadrFormX = self._gradQuadrFormX(hyperparams,dKx,columns=False)
                dlMl[:,i]=.5*(dgradLogdetX+dgradQuadrFormX)

            if VERBOSE:
                [K,Ki] = getK_verbose(KV)
                print "costly checking X gradients"
                dlMl_ = SP.zeros_like(self.x_r)
                #row covariance:
                yf = self.y.ravel()
                for i in xrange(len(self.gplvm_dimensions_r)):
                    d = self.gplvm_dimensions_r[i]
                    #K(X,X') , derivarive wr.t. first argument for dimension d
                    dKx = self.covar_r.Kgrad_x(hyperparams['covar_r'], self.x_r, self.x_r, d)
                    #K(X,X), derivartive w.r.t X, diagonal
                    for n in xrange(self.n):
                        #create covarinace derivative w.r.t to a single parameter
                        dKxn = SP.zeros([self.n, self.n])
                        dKxn[n, :]  = dKx[n, :]
                        dKxn[:, n] += dKx[n, :]
                        #1. logdet term
                        dKl = SP.kron(dKxn,KV['Kc'])
                        grad_logdet_ = 0.5 * SP.dot(Ki,dKl).trace()
                        #2. quadratic part
                        dKq = SP.dot(SP.dot(Ki,dKl),Ki)
                        grad_quad_ = - 0.5* SP.dot(SP.dot(yf,dKq),yf)
                        dlMl_[n, i] = (grad_logdet_ + grad_quad_)
                    #TODO: here do it more efficiently for comparison
                #CHECK FOR CONSISTENCY
                check_dist(dlMl,dlMl_)
                
            RV['x_r']=dlMl
        if 'x_c' in hyperparams:
            dlMl = SP.zeros_like(self.x_c)
            #cycle through GPLVM data dimensions
            for i in xrange(len(self.gplvm_dimensions_c)):
                #note, not all dimensions may be learnt, hence gplvm_dimeensions index
                d = self.gplvm_dimensions_c[i]
                #K(X,X') , derivarive wr.t. first argument for dimension d
                dKx = self.covar_c.Kgrad_x(hyperparams['covar_c'], self.x_c, self.x_c, d)
                #print ('check_dist in _LML_grad_x')
                #check_dist(dKx[0,:],dKx[1,:])
                #pdb.set_trace()
                dgradLogdetX = self._gradLogDetX(hyperparams,dKx,columns=True)
                dgradQuadrFormX = self._gradQuadrFormX(hyperparams,dKx,columns=True)
                dlMl[:,i]=.5*(dgradLogdetX+dgradQuadrFormX)
                #pdb.set_trace()
            RV['x_c'] = dlMl
        return RV



    #### derivatives w.r.t. to kernel parameters #####
    def _gradLogDet(self, hyperparams,dK,columns =False ):
        """gradient of logdet w.r.t. kernel derivative matrix (dK)"""
        KV = self.get_covariances(hyperparams)
        Si = KV['Si']
        if columns:
            d=(KV['Uc']*SP.dot(dK,KV['Uc'])).sum(0)
            RV = SP.dot(KV['Sr'],SP.dot(Si,d))
            if VERBOSE:
                #kd = SP.kron(KV['Sr'],d)
                kd = krondiag(KV['Sr'],d)
                #kd = krondiag_(KV['Sr'],d)
                RV_=SP.sum(kd*Si)
                check_dist(RV,RV_)
        else:
            #d=SP.dot(KV['Ur'].T,SP.dot(dK,KV['Ur'])).diagonal()
            d=(KV['Ur']*SP.dot(dK,KV['Ur'])).sum(0)
            RV = SP.dot(d,SP.dot(Si,KV['Sc']))
            if VERBOSE:
                #kd = SP.kron(d,KV['Sc'])
                kd=krondiag(d,KV['Sc'])
                #kd=krondiag_(d,KV['Sc'])
                RV_=SP.sum(kd*Si)
                check_dist(RV,RV_)
        return  RV

    def _gradQuadrForm(self, hyperparams,dK,columns =True ):
        """derivative of the quadtratic form w.r.t. kernel derivative matrix (dK)"""
        KV = self.get_covariances(hyperparams)
        Si = KV['Si']
        Ytilde = (KV['YSi'])
        if columns:
            UdKU = SP.dot(KV['Uc'].T,SP.dot(dK,KV['Uc']))
            SYUdKU = SP.dot((Ytilde*SP.tile(KV['Sr'][:,SP.newaxis],(1,Ytilde.shape[1]))),UdKU.T)
        else:
            UdKU = SP.dot(KV['Ur'].T,SP.dot(dK,KV['Ur']))
            SYUdKU = SP.dot(UdKU,(Ytilde*SP.tile(KV['Sc'][SP.newaxis,:],(Ytilde.shape[0],1))))
        return -SP.dot(Ytilde.ravel(),SYUdKU.ravel())

    #### derivatives w.r.t. to kernel inputs X #####
    def _gradQuadrFormX(self, hyperparams,dKx,columns =True ):
        """derivative of the quadtratic form with.r.t. covarianceparameters for row or column covariance"""
        KV = self.get_covariances(hyperparams)
        Ytilde = (KV['YSi'])
        if columns:
            UY=SP.dot(KV['Uc'],Ytilde.T)
            UYS = UY*SP.tile(KV['Sr'][SP.newaxis,:],(Ytilde.shape[1],1))
        else:
            UY=SP.dot(KV['Ur'],Ytilde)
            UYS = UY*SP.tile(KV['Sc'][SP.newaxis,:],(Ytilde.shape[0],1))
        UYSYU=SP.dot(UYS,UY.T)
        trUYSYUdK=(UYSYU*dKx.T).sum(0)
        return -2.0*trUYSYUdK

    def _gradLogDetX(self,hyperparams,dKx,columns = False):
        """calc grad log det for arbitrary X dimension
        dKx: n x n matrix, containing the derivatives of K(X1,X2) w.r.t. x_{n,:} in rows
        """
        KV = self.get_covariances(hyperparams)
        Si = KV['Si']
        if columns:
            #see gradLogDet, here we do the same computation for all row in parallel

            n = dKx.shape[0]
            D = 2*KV['Uc'] * SP.dot(dKx,KV['Uc'])
            RV = SP.dot(SP.dot(KV['Sr'],Si ),D.T)
            if VERBOSE:
                print "checking gradLogDetX"
                d1=SP.zeros(D.shape)
                d2=SP.zeros(D.shape)
                d1_=SP.zeros(D.shape)
                d2_=SP.zeros(D.shape)
                RV_= SP.zeros(D.shape[0])
                for i in SP.arange(D.shape[0]):
                    dKxn = SP.zeros([n,n])
                    dKxn[i, :] += dKx[i, :]
                    d1[i,:]=2*SP.dot(KV['Uc'].T,SP.dot(dKxn,KV['Uc'])).diagonal()
                    d2[i,:]=2*SP.dot(KV['Uc'].T,SP.dot(dKxn.T,KV['Uc'])).diagonal()
                    d1_[i,:]=2*(KV['Uc']*SP.dot(dKxn,KV['Uc'])).sum(0)
                    d2_[i,:]=2*(KV['Uc']*SP.dot(dKxn.T,KV['Uc'])).sum(0)
                    kdi=SP.kron(KV['Sc'],d2_[i,:])
                    RV_[i]=SP.dot(kdi,Si.ravel())
                check_dist(D,d1)
                check_dist(D,d2)
                check_dist(D,d1_)
                check_dist(D,d2_)
                check_dist(RV,RV_)
                kd = SP.krondiag(KV['Sr'],D)
                RV__=SP.dot(kd,Si.ravel())
                check_dist(RV,RV__)
                check_dist(RV__,RV_)
            #print "check that the kronecker product does the right thing here. it might mix different gradients"
        else:
            n = dKx.shape[0]
            D = 2*KV['Ur'] * SP.dot(dKx,KV['Ur'])
            RV=SP.dot(D,SP.dot(Si,KV['Sc']))
            #pdb.set_trace()
            if VERBOSE:
                print "checking gradLogDetX"
                d1=SP.zeros(D.shape)
                d2=SP.zeros(D.shape)
                d1_=SP.zeros(D.shape)
                d2_=SP.zeros(D.shape)
                RV_= SP.zeros(D.shape[0])
                kd = SP.kron(D,KV['Sc'])
                RV__=SP.dot(kd,Si.ravel())
                for i in SP.arange(D.shape[0]):
                    dKxn = SP.zeros([n,n])
                    dKxn[i, :] += dKx[i, :]
                    d1[i,:]=2*SP.dot(KV['Ur'].T,SP.dot(dKxn,KV['Ur'])).diagonal()
                    d2[i,:]=2*SP.dot(KV['Ur'].T,SP.dot(dKxn.T,KV['Ur'])).diagonal()
                    d1_[i,:]=2*(KV['Ur']*SP.dot(dKxn,KV['Ur'])).sum(0)
                    d2_[i,:]=2*(KV['Ur']*SP.dot(dKxn.T,KV['Ur'])).sum(0)
                    kdi=SP.kron(d2_[i,:],KV['Sc'])
                    RV_[i]=SP.dot(kdi,Si.ravel())
                check_dist(D,d1)
                check_dist(D,d2)
                check_dist(D,d1_)
                check_dist(D,d2_)
                check_dist(RV,RV_)
                
                check_dist(RV__,RV_)
                check_dist(RV,RV__)
                #pdb.set_trace()
        return RV

