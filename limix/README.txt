***********************************
TODO list
***********************************
- Check standard errrors handling (D)
- Test a bit the code (D: OK)
- Implement Unitests (D: OK)
- Logging System (D)
- move basic classes from utils to types (D, OK)
- get rid of position specificity [not using sys.path.append(...)](D, OK)
- Variance deocmpositaion module (P)
- Kronecker Covariance (P)
...
- Better Handling of Params (P)


***************************
0) Files that are in limix-dev at the moment
***************************
    - gp/gp_base.py
    - mean/mean_base.py
    - covar/combinators.py
    - covar/sqexp.py
    - covar/fixed.py
    - covar/cov_reml.py
    - optimize/optimize_bfgs_new.py
    - test/test_gp_base.py

    The rest is used by Christoph too!

**************************
1) Standard errors
**************************
    a) fisher mean is just Areml
    b) fisher covar is calculated in covar and NOT cached.
    c) after optimization if calc_ste==True gp get mean and covar Fisher's, compute inverses
    d) the gp also gives the corresponding outcomes to mean and covar
    e) covariance and mean get FIinv and cache them
    f) in particular, the covariance combinator caches the whole FIinv but also passes down specific bocks to single term covariances
        (alternatively, should it cache only external blocks?)
    TODO:
        - either solve synchronization
        - or not considering steps d), e) and f)

*****************************
2) Better handling of parameters
*****************************
    - we could name all paramers using a scheme as proposed in the following example:

        covar1 = sqexp(X,Xstar=Xstar)
        covar2 = fixed(sp.eye(N))
        covar  = sumcov(covar1,covar2)

        covar.getParamNames()
        SP.array(['sumcov.covar1_sqexp.scale',
                    'sumcov.covar1_sqexp.length',
                    'sumcov.covar2_fixed.scale'])

    - parameters can then be a panda dataframe (a vector with labels)

    - the inverse of the fisher information matrices cached in teh covariances can also be a panda dataframe (it would be much easier to read)

    - paramMask is missing at the moment
        - it should be handled in single term covariances (and mean) and propagate up to the gp
        - the argument Ifilter that now the optimizer has should be removed

    - getParams -> getParamArray??

*******************************
3) Handling GPLVM (to discuss all together)
*********************************

    - discuss about Y/y, B/b in mean (gplvm)
    - params handling not solved

***********************************
4) X handling in combinator (to discuss all together)
***********************************
    - as combinator has to have X then I think
        covar_base should have X too
        (by default the covariance does not have X)
    - Proposal
        - covar_base has property X=None
        - there is a covar_inputs with proper set and get X
    - how to handle gradient with respect to X?
