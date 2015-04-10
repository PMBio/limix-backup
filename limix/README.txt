mean_base
- add predicitons out of sample
    - how to cache? - Ystar(self,Fstar=None)
- params  handling (scheme that handles warping parameters)
    - (mean knows the mapping param_type to params - as covariance??)
        (mapping has to be known within child objects (covar, mean)
            otherwise combinators are hard to implement!)
    - gradient handling as well!!
    - how to do caching with gradients
- create method for both Y and y; B and b
    - (Y and y are matrix and vector representation of the same thing)
- handle ParamMask

gp_base
- general gp with gradients
- general gp with linear mean (and reml)
    - calculate B and set it to params
- move Areml_inv.chol,.inv,.logdet to matrix class caching?
    - Areml is in principle a suitable covariance matrix
- _grad_idx, good solution?
- paramMask into gp

covariance (->covar_base.py???)
- parameters handling
    - made proposal
    - improve define names of properties which are parameters in the child covariance class
    - getParams and setParams is in the covariance and performs the mapping from array to properties
    - handling different parametrization
- logdet gradient, how to calc it?
    - self.Kinv_dot(K_grad_i).diagonal().sum()
    - (self.Kinv()*K_grad_i()).sum()
        where self.Kinv_dot(SP.eye(N)) 
- define different parameters type
    - e.g. scale parameters does not change logdet, eigenval decomp, etc...
    - covariance knows the mapping param_type to params
        (mapping has to be known within child objects (covar, mean)
            otherwise combinators are hard to implement!)
