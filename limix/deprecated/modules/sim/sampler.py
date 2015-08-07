import numpy as np
import scipy as sp
import scipy.stats
from numpy import dot

"""Sample normally distruted effects (e.g., genetic effects) given the
total variance and the causal features.

Let $z = \sum_{i=1}^p x_i u_i$ assuming that $E[x_i] = 0$, $E[x_i^2] = 1$, and
$x_1, x_2, \dots, x_p$ are independent from each other. We want to sample
the iid $u_1, u_2, \dots, u_p$ from a normal distribution such that
$E[z] = 0$ and $E[z^2] = \sigma^2$.

Args:
    neffects: Number of effect sizes.
    causal_indices: Indices of the causal features.
    var: Total variance $\sigma^2$.

Returns:
    Sampled normally distributed effects $u$.


"""

def standardize_design(G, mean_var=None):
    if mean_var is None:
        mean_var = (0., 1./G.shape[1])
        np.apply_along_axis(lambda x: _change_sample_stats(x, (0., 1.)), 0, G)
    else:
        G -= mean_var[0]
        G /= np.sqrt(mean_var[1])

def _sample_causal_indices(size, ncausal):
    causal_indices = np.random.choice(size, size=int(ncausal), replace=False)
    return np.asarray(causal_indices, int)


def _change_sample_stats(x, mean_var=(None, None)):
    if mean_var[0] is not None:
        x -= np.mean(x)
        x += mean_var[0]

    if mean_var[1] is not None:
        v = np.std(x) if len(x) > 1 else x[0]
        x /= v
        x *= np.sqrt(mean_var[1])

def static_effsiz_sampler(effects):
    def sampler(n):
        return effects
    return sampler

def normal_effsiz_sampler():
    def sampler(n):
        if n == 0:
            return np.empty(0, float)
        effects = np.random.randn(n)
        return effects
    return sampler

def binary_effsiz_sampler():
    def sampler(n):
        if n == 0:
            return np.empty(0, float)
        effects = np.random.randn(n)
        effects[effects >= 0] = +1.
        effects[effects <  0] = -1.
        return effects
    return sampler

class TraitSampler(object):
    def __init__(self):
        self._Gs = dict()
        self._zc = dict()
        self._causal_indices = dict()
        self._us = dict()
        self._eff_sample_mean_vars = dict()
        self._noise_var = None

    def add_effects_design(self, name, G, effsiz_sampler,
                           effsiz_sample_mean_var=None,
                           eff_sample_mean_var=None, ncausal=None):

        if ncausal is None:
            ncausal = G.shape[1]

        self._Gs[name] = G

        causal_indices = _sample_causal_indices(G.shape[1], ncausal)
        self._causal_indices[name] = causal_indices

        u = effsiz_sampler(len(causal_indices))

        if effsiz_sample_mean_var is not None:
            (m, v) = effsiz_sample_mean_var
            _change_sample_stats(u, mean_var=(m, v))

        self._us[name] = u
        self._eff_sample_mean_vars[name] = eff_sample_mean_var

    def add_effect_cov(self, name, K, effsize_sample_mean_var=None):
        zeros = np.zeros(K.shape[0])
        if effsize_sample_mean_var is None:
            zc = sp.stats.multivariate_normal(zeros, K).rvs()
        else:
            L = np.linalg.cholesky(K)
            u = np.random.randn(K.shape[0])
            m = effsize_sample_mean_var[0]
            v = effsize_sample_mean_var[1]
            _change_sample_stats(u, (0., v))
            zc = dot(L.T, u)

        self._zc[name] = zc

    def set_noise(self, var):
        self._noise_var = var

    def sample_traits(self):
        z = 0
        n = self._Gs.values()[0].shape[0]
        ze = np.random.randn(n) * np.sqrt(self._noise_var)
        z = ze
        zd = dict()
        for effect_name in self._Gs.keys():
            G = self._Gs.pop(effect_name)
            u = self._us[effect_name]
            idx = self._causal_indices[effect_name]
            zd[effect_name] = dot(G[:, idx], u)

            if self._eff_sample_mean_vars[effect_name] is not None:
                zd[effect_name] = _change_sample_stats(zd[effect_name],
                                               self._eff_sample_mean_vars)

            z += zd[effect_name]

        for z_ in self._zc.values():
            z += z_

        return (z, zd, self._zc, ze)

# if __name__ == '__main__':
#     neffects = 30
#     causal_indices = np.random.choice(neffects, 15)
#     var = 5.0
#     u = sample_effects(neffects, causal_indices, 'Normal', var)
#     print u
#     print np.sum(u[causal_indices]**2)
#
#     u = sample_effects(neffects, causal_indices, 'Binary', var)
#     print u
#     print np.sum(u[causal_indices]**2)

if __name__ == '__main__':
    from dreader import DReader1000G
    from dreader import normalize_genotype
    import h5py
    ncovariates = 1
    nindividuals = 1000
    # nparents = 2
    #
    # fore_chroms = [20]
    # back_chroms = [21]
    # back2_chroms = [22]
    # pops = ['EUR']
    # maf = 0.05
    #
    # with DReader1000G('/Users/horta/workspace/1000G_majors_c.hdf5',
    # # with DReader1000G('/Users/horta/workspace/1000G_fake.hdf5',
    #                   maf, pops) as dr:
    #
    #     fore_mean_std = dr.mean_std(fore_chroms)
    #     back_mean_std = dr.mean_std(back_chroms)
    #     back2_mean_std = dr.mean_std(back2_chroms)
    #
    #     fore_genos = []
    #     back_genos = []
    #     back2_genos = []
    #     covariatess = []
    #     for i in xrange(nindividuals):
    #         parents = dr.choose_parents(nparents)
    #         covariate = np.random.randn(ncovariates)
    #         fore_geno = dr.generate_genotype(parents, fore_chroms)
    #         back_geno = dr.generate_genotype(parents, back_chroms)
    #         back2_geno = dr.generate_genotype(parents, back2_chroms)
    #
    #         covariatess.append(covariate)
    #         fore_genos.append(fore_geno)
    #         back_genos.append(back_geno)
    #         back2_genos.append(back2_geno)
    #
    #     fG = np.array(fore_genos, float)
    #     bG = np.array(back_genos, float)
    #     b2G = np.array(back2_genos, float)
    #     nb2G = standardize_genotype(b2G, back2_mean_std)
    #     b2K = dot(nb2G, nb2G.T) / nb2G.shape[1]
    #
    #
    #     X = np.array(covariatess, float)
    #
    #     def create_dataset(grp, name, M):
    #         grp.create_dataset(name, chunks=(1, M.shape[1]),
    #                            compression='lzf', data=M)
    #
    #     with h5py.File("/Users/horta/workspace/sample1000.hdf5", "w") as f:
    #         ggrp = f.create_group("genotypes")
    #
    #         create_dataset(ggrp, "fG_inds_by_snps", fG)
    #         create_dataset(ggrp, "fG_snps_by_inds", fG.T)
    #         ggrp.create_dataset("fore_mean_std", data=np.asarray(fore_mean_std))
    #
    #         create_dataset(ggrp, "bG_inds_by_snps", bG)
    #         create_dataset(ggrp, "bG_snps_by_inds", bG.T)
    #         ggrp.create_dataset("back_mean_std", data=np.asarray(back_mean_std))
    #
    #         ggrp.create_dataset("b2K", data=b2K)
    #
    #         cgrp = f.create_group("covariates")
    #         cgrp.create_dataset("X", data=X)

    fore_var = 0.4
    back_var = 0.4
    noise_var = 0.2
    nsnps = 5000

    X = np.ones((nindividuals, 1))
    beta = np.zeros(X.shape[1])

    pop_means = np.random.randn(nsnps) * 0.1
    pop_vars = np.random.gamma(1, size=nsnps) * 0.1
    G = np.random.randn(nindividuals, nsnps)*np.sqrt(pop_vars) + pop_means

    K = np.random.randn(nindividuals, nindividuals)
    K = dot(K, K.T)

    ns = TraitSampler()

    ns.add_effects_design("covariates", X, static_effsiz_sampler(beta),
                          effsiz_sample_mean_var=None, # default
                          eff_sample_mean_var=None, # default
                          ncausal=None) # default

    ncausal = nsnps
    standardize_design(G, (pop_means, pop_vars))

    ns.add_effects_design("foreground", G, binary_effsiz_sampler(),
                          effsiz_sample_mean_var=(0., fore_var/float(ncausal)),
                          eff_sample_mean_var=None, # default
                          ncausal=ncausal)

    from limix.utils.preprocess import covar_rescale
    K = covar_rescale(K)
    ns.add_effect_cov("background", K, (0., back_var))
    #
    ns.set_noise(noise_var)
    (y, zd, zc, ze) = ns.sample_traits()
    #
    print np.var(y)
    # print np.var(zd['foreground'])
