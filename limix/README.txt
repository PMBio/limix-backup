TODO:
- fix the gp (P)
- cov and mean: two inner prediction methods for prediction
(in-of-sample cached,out-of-sample not) (P)
- gp/mean: change Y/B to y/b (P)
- optimization + standard error

TODO next (discussed):
    - discuss about Y/y, B/b in mean (gplvm)
    - params handling not solved
    - getParams -> getParamArray

mean_base
- add predicitons out of sample
    - how to cache? - Ystar(self,Fstar=None)
- create method for both Y and y; B and b
    - (Y and y are matrix and vector representation of the same thing)
- handle paramMask

gp_base
- general gp with gradients
- general gp with linear mean (and reml)
    - calculate B and set it to params
- move Areml_inv.chol,.inv,.logdet to matrix class caching?
    - Areml is in principle a suitable covariance matrix
- _grad_idx, good solution?

covariance (->covar_base.py???)
- specialized covar knows parameters with proper name,
    getParams build the vactor on spot.
    Same thing in combinators and gp
- alternatively vector of params is known and specific setters act directely on the elements of the array
- paramMask into covar
- caching for combinators does not work properly
    - combinators has to know if some parameter has changed 
    - reimplement ideas in LIMIX for caching?
    - handling different parametrization
- logdet gradient, how to calc it?
    - self.Kinv_dot(K_grad_i).diagonal().sum()
    - (self.Kinv()*K_grad_i()).sum()
        where self.Kinv_dot(SP.eye(N)) 
- scale parameters does not change logdet, eigenval decomp, etc...
