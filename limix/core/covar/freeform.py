import scipy as SP
from covariance import covariance
import pdb

class freeform(covariance):
    """
    freeform covariance function
    """
    def __init__(self,P):
        """
        initialization
        """
        covariance.__init__(self,P)
        self.L = SP.zeros((self.P,self.P))
        self.Lgrad = SP.zeros((self.P,self.P))
        self.zeros = SP.zeros(self.n_params)

    def K(self):
        """
        evaluates the kernel for given hyperparameters theta
        """
        self._updateL()
        RV = SP.dot(self.L,self.L.T)
        return RV
     
    def Kgrad_param(self,i):
        """
        partial derivative with repspect to the i-th hyperparamter theta[i]
        """
        self._updateL()
        self._updateLgrad(i)
        RV = SP.dot(self.L,self.Lgrad.T)+SP.dot(self.Lgrad,self.L.T)
        return RV 

    def _calcNumberParams(self):
        """
        calculates the number of parameters
        """
        self.n_params = int(0.5*self.P*(self.P+1))

    def _updateL(self):
        """
        construct the cholesky factor from hyperparameters
        """
        self.L[SP.tril_indices(self.P)] = self.params

    def _updateLgrad(self,i):
        """
        construct the cholesky factor from hyperparameters
        """
        self.zeros[i] = 1
        self.Lgrad[SP.tril_indices(self.P)] = self.zeros
        self.zeros[i] = 0

