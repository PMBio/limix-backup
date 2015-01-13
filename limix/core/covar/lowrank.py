import scipy as SP
from covariance import covariance

class lowrank(covariance):
    """
    lowrank covariance
    """
    def __init__(self,P,rank=1):
        """
        initialization
        """
        self.rank = rank
        covariance.__init__(self,P)
        
    def K(self):
        """
        evaluates the kernel for given hyperparameters theta
        """
        X = self._getX()
        RV = SP.dot(X,X.T)
        return RV
     
    def Kgrad_param(self,i):
        """
        partial derivative with repspect to the i-th hyperparamter theta[i]
        """
        X = self._getX()
        Xgrad = self._getXgrad(i)
        import pdb
        RV = SP.dot(X,Xgrad.T)+SP.dot(Xgrad,X.T)
        return RV 

    def _calcNumberParams(self):
        """
        calculates the number of parameters
        """
        self.n_params = int(self.P*self.rank)

    def _getX(self):
        """
        reshape the parameters
        """
        RV = SP.reshape(self.params,(self.P,self.rank),order='F')
        return RV

    def _getXgrad(self,i):
        """
        reshape the parameters
        """
        zeros = SP.zeros(self.n_params)
        zeros[i] = 1
        RV = SP.reshape(zeros,(self.P,self.rank),order='F')
        return RV

