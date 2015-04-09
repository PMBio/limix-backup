mean_base
- add predicitons out of sample
    - how to cache? - Ystar(self,Fstar=None)
- params  handling (scheme that handles warping parameters)
    - (mean knows the mapping param_type to params - as covariance??)
        (mapping has to be known within child objects (covar, mean)
            otherwise combinators are hard to implement!)
    - gradient handling as well!!
    - how to do caching with gradients

gp_base
- general gp with gradients
- general gp with linear mean (and reml)
    - calculate B and set it to params
- move Areml_inv.chol,.inv,.logdet to matrix class caching?

covariance (->covar_base.py???)
- define different parameters type
    - e.g. scale parameters does not change logdet, eigenval decomp, etc...
    - covariance knows the mapping param_type to params
        (mapping has to be known within child objects (covar, mean)
            otherwise combinators are hard to implement!)
