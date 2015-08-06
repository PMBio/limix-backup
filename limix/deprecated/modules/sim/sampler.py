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
        # random effect variances
        self._vars = dict()
        # fixed effects
        self._beta = None
        # random effects
        self._us = dict()
        self._covariates = None
        self._noise_var = None

    def set_fixed_effects(self, beta):
        self._beta = np.asarray(beta, dtype=float)

    def add_random_effects(self, name, var, nsnps, frac_causal):

        if frac_causal == 'all':
            causal_indices = np.arange(nfeatures, dtype=int)
        else:
            causal_indices = np.random.choice(nfeatures,
                                              size=int(nfeatures * frac_causal),
                                              replace=False)
            causal_indices = np.asarray(causal_indices, int)

        self._causal_indices = causal_indices
        self._vars[name] = var
        self._us[name] = sample_effects(nfeatures, causal_indices, var)
        return causal_indices

    def set_covariates(self, covariates):
        covariates = np.asarray(covariates, float)
        self._covariates = covariates

    def set_noise(self, var):
        self._noise_var = var

    def set_features(self, name, features):
        self._features[name] = features

    def sample_trait(self):
        assert self._beta is not None,
            "You have to set the fixed effects first."\
        assert self._covariates is not None,\
            "You have to set the covariates first."

        z = 0
        zf = dot(self._covariates, self._beta)
        zr = dict()
        for (fname, ffeatures) in self._features.items():
            var = self._vars[fname]
            features = self._features.pop(fname)
            u = self._us[fname]
            zr[fname] = dot(features, u)

        ze = np.random.randn() * np.sqrt(self._noise_var)
        z = zf + np.sum(zr.values()) + ze
        return (z, zf, zr, ze)

if __name__ == '__main__':
    from dreader import DReader1000G
    from dreader import normalize_genotype
    nindividuals = 10000
    nparents = 5
    beta = [0.]
    fore_var = 0.4
    back_var = 0.4
    noise_var = 0.2
    fore_chroms = [1, 2]
    back_chroms = [20, 21]
    pops = ['EUR']
    maf = 0.05
    fore_frac_causal = 0.5
    back_frac_causal = 0.5
    y = []
    # with DReader1000G('/Users/horta/workspace/1000G_majors_c.hdf5',
    with DReader1000G('/Users/horta/workspace/1000G_fake.hdf5',
                      maf, pops) as dr:
        ns = NormalSampler()

        ns.set_fixed_effects(beta)

        # FOREGROUND
        nsnps = dr.nsnps(fore_chroms)
        fore_causal_indices =\
            ns.add_random_effects("foreground", fore_var, nsnps,
                                  fore_frac_causal)
        fore_mean_std = dr.mean_std(fore_chroms)

        # BACKGROUND
        nsnps = dr.nsnps(back_chroms)
        back_causal_indices =\
            ns.add_random_effects("background", back_var, nsnps,
                                  back_frac_causal)
        back_mean_std = dr.mean_std(back_chroms)

        ns.set_noise(noise_var)

        for i in xrange(nindividuals):
            parents = dr.choose_parents(nparents)

            fore_geno = dr.generate_genotype(parents, fore_chroms)
            back_geno = dr.generate_genotype(parents, back_chroms)

            ns.set_covariates(np.random.randn(len(beta)))

            g = normalize_genotype(fore_geno, fore_mean_std)
            ns.set_features("foreground", g)
            g = normalize_genotype(back_geno, back_mean_std)
            ns.set_features("background", g)

            (z, zf, zr, ze) = ns.sample_trait()
            y.append(z)

    y = np.asarray(y, float)
    print np.mean(y)
    print np.var(y)
