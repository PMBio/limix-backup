from covariance import covariance
import pdb
import scipy as SP

class sumcov(covariance):

    def __init__(self,*covars):
        self.dim = None
        self.covars = []
        for covar in covars:
            self.addCovariance(covar)
        self._grad_idx = 0
        self.clear_all()
        
    #####################
    # Covars handling
    #####################
    def addCovariance(self,covar):
        if self.dim is None:
            self.dim = covar.dim
        else:
            assert covar.dim==self.dim, 'Dimension mismatch'
        self.covars.append(covar)
        self._calcNumberParams()
        
    def getCovariance(self,i):
        return self.covars[i]

    #####################
    # Params handling
    #####################
    def setParams(self,params):
        istart = 0
        for i in range(len(self.covars)):
            istop = istart + self.getCovariance(i).getNumberParams()
            self.getCovariance(i).setParams(params[istart:istop])
            istart = istop
        self.clear_all()

    def getParams(self):
        istart = 0
        params = SP.zeros(self.getNumberParams())
        for i in range(len(self.covars)):
            istop = istart + self.getCovariance(i).getNumberParams()
            params[istart:istop] = self.getCovariance(i).getParams()
            istart = istop
        return params
    
    #####################
    # Cached
    #####################
    """
    The caching for now does not work
    need to talk to single covariance matrices
    """
    def K(self):
        K = SP.zeros((self.dim,self.dim))
        for i in range(len(self.covars)):
            K += self.getCovariance(i).K()
        return K

    #def Kcross(self):
    #    """
    #    evaluates the kernel between test and training points for given hyperparameters
    #    """
    #    n_test  = self.Xstar.shape[0]
    #    n_train = self.X.shape[0]
    #    Kcross  = SP.zeros((n_test,n_train))
    #
    #    for i in range(len(self.covars)):
    #        Kcross += self.covars[i].Kcross()
    #    
    #    return Kcross
        
    def K_grad_i(self):
        istart = 0
        for j in range(len(self.covars)):
            istop = istart + self.getCovariance(j).getNumberParams()
            if (self._grad_idx < istop): 
                idx = self._grad_idx - istart 
                self.getCovariance(j).set_grad_idx(idx)
                return self.getCovariance(j).K_grad_i()
            istart = istop 
        return None
    
    def _calcNumberParams(self):
        self.n_params = 0
        for i in range(len(self.covars)):
            self.n_params += self.getCovariance(i).getNumberParams()
        return self.n_params
            
    ####################
    # DEPRECATED FUNCTIONS
    ####################
    #def setX(self,X):
    #    self.X = X
    #    for i in range(len(self.covars)):
    #        self.covars[i].setX(X)
    #def setXstar(self,Xstar):
    #    self.Xstar = Xstar
    #    for i in range(len(self.covars)):
    #        self.covars[i].setXstar(Xstar)


