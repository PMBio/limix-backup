import numpy as np

def standardize_genotype(genotype, mean_std=None):
    if mean_std is None:
        m = np.mean(genotype, axis=0)
        v = np.std(genotype, axis=0)
        v[v == 0.] = 1.0
    else:
        m = mean_std[0]
        v = mean_std[1]
    return (genotype - m) / v

def standardize_covariance(K):
    from limix.utils.preprocess import covar_rescale
    return covar_rescale(K)
