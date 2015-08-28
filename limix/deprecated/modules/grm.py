import numpy as np
import scipy as sp
import scipy.stats

def _calculate_number_alleles(G):
    G = np.asarray(G, int)
    assert len(G.shape) == 2
    u = np.unique(G[:])
    assert np.all([ui in [0, 1, 2] for ui in u])

    b = np.sum(G, axis=0)
    a = G.shape[0]*2 - b

    return (a, b)

def _normalize_maf_allele(G):
    (a, b) = _calculate_number_alleles(G)
    change = b > a
    G[:, change] = 2 - G[:, change]

def _calculate_maf(G):
    return np.sum(G, 0) / float(2*G.shape[0])

# According to GCTA's paper
def grm_unbiased(G):
    _normalize_maf_allele(G)
    p = _calculate_maf(G)
    n = G.shape[0]

    K = np.zeros((n, n))

    for j in xrange(n-1):
        for k in xrange(j+1, n):

            v0 = (G[j, :] - 2*p) * (G[k, :] - 2*p)
            v1 = 2 * p * (1. - p)
            np.mean(v0 / v1)

            K[j, k] = np.mean(v0 / v1)
            K[k, j] = K[j, k]

    for j in xrange(n):
        g2 = G[j, :]**2
        v0 = g2 - (1 + 2*p) * G[j, :] + 2 * p**2
        v1 = 2 * p * (1. - p)
        K[j, j] = 1 + np.mean(v0 / v1)

    return K

if __name__ == '__main__':
    np.random.seed(5)
    # G = np.random.randint(0, 3, (100000, 1))
    G = sp.stats.binom.rvs(2, 0.5, size=(1000, 10))
    # _calculate_maf(G)
    K = grm_unbiased(G)
    import ipdb; ipdb.set_trace()
    print K
