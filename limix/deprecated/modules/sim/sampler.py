import numpy as np
import scipy as sp
import scipy.stats
from numpy import dot
from limix.core.linalg.linalg_matrix import QS_from_K
import sys

def standardize_design(G, mean_var=None):
    if mean_var is None:
        mean_var = (0., 1./G.shape[1])
        np.apply_along_axis(lambda x: _change_sample_stats(x, (0., 1.)), 0, G)
    else:
        G -= mean_var[0]
        G /= np.sqrt(mean_var[1])

def standardize_cov(K):
    from limix.utils.preprocess import covar_rescale
    return covar_rescale(K)

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

def geometric_ntrials_sampler(min_ntrials, max_ntrials, p):
    def sampler():
        k = np.random.geometric(p) + min_ntrials - 1
        while k >= max_ntrials:
            k = np.random.geometric(p) + min_ntrials - 1
        return k
    return sampler

class TraitSampler(object):
    def __init__(self):
        self._Gs = dict()
        self._zc = dict()
        self._causal_indices = dict()
        self._us = dict()
        self._eff_sample_mean_vars = dict()
        self._noise_var = 0.
        self._nindividuals = None

    def add_effects_design(self, name, G, effsiz_sampler,
                           effsiz_sample_mean_var=None,
                           eff_sample_mean_var=None,
                           ncausal=None,
                           min_dist=1,
                           base_pos=None):

        if ncausal is None:
            ncausal = G.shape[1]

        self._Gs[name] = G

        if base_pos is None:
            causal_indices = self._sample_causal_indices(G.shape[1], ncausal)
        else:
            causal_indices = self._sample_causal_indices_pos(base_pos, ncausal,
                                                             min_dist)
        self._causal_indices[name] = causal_indices

        u = effsiz_sampler(len(causal_indices))

        if effsiz_sample_mean_var is not None:
            (m, v) = effsiz_sample_mean_var
            _change_sample_stats(u, mean_var=(m, v))

        self._us[name] = u
        self._eff_sample_mean_vars[name] = eff_sample_mean_var

        if self._nindividuals is None:
            self._nindividuals = G.shape[0]

    def causal(self, name):
        return self._causal_indices[name]

    def add_effect_cov(self, name, K=None, effsize_sample_mean_var=None,
                       Q=None, S=None):
        import scipy as sp
        import scipy.stats
        if Q is None or S is None:
            (Q, S) = QS_from_K(K)
        S = np.sqrt(S)

        u = np.random.randn(S.shape[0])
        if effsize_sample_mean_var is not None:
            m = effsize_sample_mean_var[0]
            v = effsize_sample_mean_var[1]
            _change_sample_stats(u, (m, v))

        self._zc[name] = dot(Q, S * u)

        if self._nindividuals is None:
            self._nindividuals = Q.shape[0]

    def set_noise(self, var):
        self._noise_var = var

    def get_noise(self):
        return self._noise_var

    def sample_traits(self):
        z = 0
        n = self._nindividuals
        ze = np.random.randn(n) * np.sqrt(self._noise_var)
        z = ze.copy()
        zd = dict()
        for effect_name in list(self._Gs.keys()):
            G = self._Gs[effect_name]
            u = self._us[effect_name]
            idx = self._causal_indices[effect_name]
            zd[effect_name] = dot(G[:, idx], u)

            if self._eff_sample_mean_vars[effect_name] is not None:
                zd[effect_name] = _change_sample_stats(zd[effect_name],
                                               self._eff_sample_mean_vars)

            z += zd[effect_name]

        for z_ in list(self._zc.values()):
            z += z_

        return (z, zd, self._zc, ze)

    def _sample_causal_indices_trial(self, base_pos, ncausal, min_dist=1):
        size = len(base_pos)
        causals = []

        failed = 0
        while len(causals) < ncausal and failed < ncausal * 100:
            i = np.random.randint(size)
            if self._distance_next_causal(i, causals, base_pos) >= min_dist:
                causals.append(i)
            else:
                failed += 1

        if len(causals) != ncausal:
            return None

        return np.array(causals, int)

    def _sample_causal_indices_pos(self, base_pos, ncausal, min_dist=1):
        while True:
            causals = self._sample_causal_indices_trial(base_pos, ncausal,
                                                        min_dist)
            if causals is not None:
                return causals

    def _sample_causal_indices(self, size, ncausal):
        return np.random.choice(size, ncausal, replace=False)

    def _distance_next_causal(self, causal, causals, base_pos):
        if len(causals) == 0:
            return np.inf
        pos = base_pos[causal]
        poss = base_pos[causals]
        return np.min(np.abs(poss - pos))

    def _zmean(self):
        assert self.get_noise() == 0.
        (z, _, _, ze) = TraitSampler.sample_traits(self)
        return np.mean(z - ze)

def _change_sample_stats(x, mean_var=(None, None)):
    if mean_var[0] is not None:
        x -= np.mean(x)
        x += mean_var[0]

    if mean_var[1] is not None:
        v0 = np.mean(x**2)
        v1 = mean_var[1]

        if v0 == 0.:
            x[:] = np.sqrt(v1)
        else:
            c = np.sqrt(v1/v0)
            x *= c

class BernoulliTraitSampler(TraitSampler):
    def __init__(self):
        TraitSampler.__init__(self)

    def _offset_due_prevalence(self, var_noise, prevalence):

        print ("Calculating offset due to prevalence...")

        (_, z) = self._sample_traits_once(var_noise, 0.)
        offset = np.percentile(z, 100. * (1-prevalence))
        print("Calculated offset: %.3f" % offset)
        return offset

    def _sample_traits_once(self, var_noise, offset):

        (z, _, _, _) = TraitSampler.sample_traits(self)
        z += np.random.randn(z.shape[0]) * np.sqrt(var_noise)
        y = np.zeros(z.shape[0], float)
        y[z >= offset] = 1.

        return (y, z)

    def sample_traits(self, pop_size, var_noise, prevalence=0.5, ascertainment=0.5):
        print("Prevalence: %.3f." % prevalence)
        print("Ascertainment: %.3f." % ascertainment)
        offset = self._offset_due_prevalence(var_noise, prevalence)

        print("Sampling traits...")
        n1 = int(ascertainment * pop_size)
        n0 = int(pop_size) - n1

        (y_, z_) = self._sample_traits_once(var_noise, offset)
        ok0 = np.where(y_ == 0.)[0]
        ok1 = np.where(y_ == 1.)[0]

        assert len(ok0) >= n0
        assert len(ok1) >= n1

        np.random.shuffle(ok0)
        np.random.shuffle(ok1)

        ok0 = ok0[:n0]
        ok1 = ok1[:n1]

        ok = np.concatenate((ok0, ok1))

        y = y_[ok]
        z = z_[ok]

        selected_individuals = ok

        print("Done.")
        return (y, z, offset, selected_individuals)

class BinomialTraitSampler(TraitSampler):
    def __init__(self):
        TraitSampler.__init__(self)

    def _sample_ntrials(self, pop_size, ntrials_sampler):
        ntrials = []
        for i in range(pop_size):
            ntrials.append(ntrials_sampler())
        return np.array(ntrials, int)

    def sample_traits(self, pop_size, vare, ntrials_sampler):

        print("Sampling traits...")

        (z, _, _, _) = TraitSampler.sample_traits(self)
        selected_individuals = np.random.choice(len(z), pop_size, replace=False)
        z = z[selected_individuals]

        ntrials = self._sample_ntrials(pop_size, ntrials_sampler)

        E = np.random.randn(pop_size, np.max(ntrials))
        E *= np.sqrt(vare)

        Z = z[:, np.newaxis] + E

        Z[Z >  0.] = 1.
        Z[Z <= 0.] = 0.

        y = np.empty(pop_size)
        for i in range(y.shape[0]):
            y[i] = np.sum(Z[i,:ntrials[i]])

        print("Done.")

        y = dict(trait=y, ntrials=ntrials)
        return (y, selected_individuals)


if __name__ == '__main__':
    pass
