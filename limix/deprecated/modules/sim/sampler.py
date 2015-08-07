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


def _standardize_effects(effects, var):
    m = np.mean(effects)
    if len(effects) > 1:
        v = np.std(effects)
    else:
        v = effects[0]
    effects -= m
    effects /= v
    effects *= np.sqrt(var) / np.sqrt(len(effects))

def static_effsiz_sampler(effects):
    def sampler(n):
        return effects
    return sampler

def normal_effsiz_sampler(var):
    def sampler(n):
        if n == 0:
            return np.empty(0, float)
        effects = np.random.randn(n)
        _standardize_effects(effects, var)
        return effects
    return sampler

def binary_effsiz_sampler(var):
    def sampler(n):
        if n == 0:
            return np.empty(0, float)
        effects = np.random.randn(n)
        effects[effects >= 0] = +1.
        effects[effects <  0] = -1.
        _standardize_effects(effects, var)
        return effects
    return sampler

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

class TraitSampler(object):
    def __init__(self):
        self._Gs = dict()
        self._zc = dict()
        self._causal_indices = dict()
        self._us = dict()
        self._noise_var = None

    def add_effects_design(self, name, effect_sampler, G, ncausal=None):

        if ncausal is None:
            ncausal = G.shape[1]

        self._Gs[name] = G
        p = G.shape[1]
        causal_indices = np.random.choice(p, size=int(ncausal),
                                          replace=False)
        causal_indices = np.asarray(causal_indices, int)

        self._causal_indices[name] = causal_indices
        self._us[name] = effect_sampler(len(causal_indices))

    def add_effect_cov(self, name, K, var=1.):
        zeros = np.zeros(K.shape[0])
        self._zc[name] = sp.stats.multivariate_normal(zeros, var * K).rvs()

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
    # nindividuals = 1000
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

    with h5py.File("/Users/horta/workspace/sample1000.hdf5", "r") as f:
        fore_var = 0.4
        back_var = 0.4
        back2_var = 1.0
        noise_var = 0.2

        X = f['covariates']['X'][:]
        beta = np.array([0.] * X.shape[1])
        genos = f['genotypes']

        fG = standardize_genotype(genos['fG_inds_by_snps'][:],
                                  genos['fore_mean_std'])

        bG = standardize_genotype(genos['bG_inds_by_snps'][:],
                                  genos['back_mean_std'][:])

        ns = TraitSampler()

        ns.add_effects_design("covariates", static_effsiz_sampler(beta), X)
        ns.add_effects_design("foreground", binary_effsiz_sampler(fore_var), fG)
        ns.add_effects_design("background", normal_effsiz_sampler(back_var), bG)

        K = genos['b2K'][:]
        ns.add_effect_cov("background2", K, back2_var)

        ns.set_noise(noise_var)
        (y, zd, zc, ze) = ns.sample_traits()

        print np.mean(y)
        print np.var(y)
