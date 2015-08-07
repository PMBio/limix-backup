import numpy as np

def standardize_design(design, mean_std=None):
    if mean_std is None:
        m = np.mean(design, axis=0)
        v = np.std(design, axis=0)
        v[v == 0.] = 1.0
    else:
        m = mean_std[0]
        v = mean_std[1]
    return (design - m) / v

def standardize_covariance(K):
    from limix.utils.preprocess import covar_rescale
    return covar_rescale(K)
