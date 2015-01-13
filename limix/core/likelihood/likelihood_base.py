import pdb
import scipy as SP

class ALik(object):
    """abstract class for arbitrary likelihood model"""
    pass

class GaussLikISO(ALik):
    """Gaussian isotropic likelihood model
    """
    def K(self,theta,n_dimensions):
        sigma = SP.exp(2*theta[0])
        Knoise = sigma*SP.eye(n_dimensions)
        return Knoise

    def Kdiag(self,theta,n_dimensions):
        sigma = SP.exp(2*theta[0])
        return sigma*SP.ones(n_dimensions)

    def Kgrad_theta(self,theta,n_dimensions,i):
        assert i==0, 'unknown hyperparameter'
        K = self.K(theta,n_dimensions)
        return 2*K

    def Kdiag_grad_theta(self,theta,n_dimensions,i):
        assert i==0, 'unknown hyperparameter'
        Kdiag = self.Kdiag(theta,n_dimensions)
        return 2*Kdiag

