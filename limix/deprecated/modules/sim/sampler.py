import numpy as np
from numpy import dot

"""Sample normally distruted effects (e.g., genetic effects) given the
total variance and the causal features.

Let $z = \sum_{i=1}^p x_i u_i$ assuming that $E[x_i] = 0$, $E[x_i^2] = 1$, and
$x_1, x_2, \dots, x_p$ are independent from each other. We want to sample
the iid $u_1, u_2, \dots, u_p$ from a normal distribution such that
$E[z] = 0$ and $E[z^2] = \sigma^2$.

Args:
    nfeatures: Number of features.
    causal_indices: Indices of the causal features.
    var: Total variance $\sigma^2$.

Returns:
    Sampled normally distributed effects $u$.


"""
def sample_effects(nfeatures, causal_indices, var):
    u = np.zeros(nfeatures)
    u[causal_indices] = np.random.randn(len(causal_indices)) * np.sqrt(var)
    u[causal_indices] /= np.sqrt(len(causal_indices))
    return u

class NormalSampler(object):
    def __init__(self):
        self._features = dict()
        self._causal_indices = dict()
        # random effects variances
        self._vars = dict()
        # fixed effects
        self._beta = None
        # random effects
        self._us = dict()
        self._covariates = None

    def set_fixed_effects(self, beta):
        self._beta = np.asarray(beta, dtype=float)

    def add_random_effects(self, name, nfeatures, var, causal_indices='all'):

        if causal_indices == 'all':
            causal_indices = np.arange(nfeatures, dtype=int)
        else:
            causal_indices = np.asarray(causal_indices, dtype=int)

        self._causal_indices = causal_indices
        self._vars[name] = var
        self._us[name] = sample_effects(nfeatures, causal_indices, var)

    def set_covariates(self, covariates):
        self._covariates = covariates

    def set_features(self, name, features):
        self._features[name] = features

    def sample_trait(self):
        z = 0
        zf = dot(self._covariates, self._beta)
        zr = dict()
        for (fname, ffeatures) in self._features.items():
            var = self._vars[fname]
            features = self._features.pop(fname)
            u = self._us[fname]
            zr[fname] = dot(features, u)

        z = zf + np.sum(zr.values())
        return (z, zf, zr)

if __name__ == '__main__':
    from dreader import DReader1000G
    from dreader import normalize_genotype
    nindividuals = 100
    nparents = 5
    beta = [0.]
    fore_var = 0.4
    back_var = 0.4
    noise_var = 0.2
    fore_nsnps = 100
    back_nsnps = 100
    fore_chroms = [1, 4]
    back_chroms = [22]
    causal_fore_snps = np.random.choice(fore_nsnps,
                                        fore_nsnps - int(0.5*fore_nsnps),
                                        replace=False)
    causal_back_snps = 'all'
    y = []
    with DReader1000G('/Users/horta/workspace/1000G_majors.hdf5') as dr:
        ns = NormalSampler()

        ns.set_fixed_effects(beta)

        # FOREGROUND
        fore_snpsref = dr.snps_reference(fore_chroms, fore_nsnps)
        ns.add_random_effects("foreground", fore_nsnps, fore_var,
                              causal_fore_snps)
        fore_mean_std = dr.mean_std(fore_snpsref, ['EUR'])

        # BACKGROUND
        back_snpsref = dr.snps_reference(back_chroms, back_nsnps)
        ns.add_random_effects("background", back_nsnps, back_var,
                              causal_back_snps)
        back_mean_std = dr.mean_std(back_snpsref, ['EUR'])

        ns.add_random_effects("noise", 1, noise_var)

        for i in xrange(nindividuals):
            parents = dr.choose_parents('EUR', nparents)

            fore_geno = dr.generate_genotype(parents, fore_snpsref)
            back_geno = dr.generate_genotype(parents, back_snpsref)

            ns.set_covariates(np.random.randn(len(beta)))

            ns.set_features("foreground", normalize_genotype(fore_geno,
                                                             fore_mean_std))
            ns.set_features("background", normalize_genotype(back_geno,
                                                             back_mean_std))
            ns.set_features("noise", [1.])

            (z, zf, zr) = ns.sample_trait()
            y.append(z)

    y = np.asarray(y)
    print np.mean(y)
