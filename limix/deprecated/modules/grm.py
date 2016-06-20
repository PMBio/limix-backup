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
    denom = 2 * p * (1. - p)
    n = G.shape[0]

    K = np.zeros((n, n))

    for j in range(n-1):
        for k in range(j+1, n):

            v0 = (G[j, :] - 2*p) * (G[k, :] - 2*p)

            K[j, k] = np.mean(v0 / denom)
            K[k, j] = K[j, k]

    for j in range(n):
        g2 = G[j, :]**2
        v0 = g2 - (1 + 2*p) * G[j, :] + 2 * p**2
        K[j, j] = 1 + np.mean(v0 / denom)

    return K

if __name__ == '__main__':
    # np.random.seed(5)
    # # G = np.random.randint(0, 3, (100000, 1))
    # G = sp.stats.binom.rvs(2, 0.5, size=(1000, 10))
    # # _calculate_maf(G)
    # K = grm_unbiased(G)
    # import ipdb; ipdb.set_trace()
    # print K


    import numpy as np
    import scipy as sp
    import scipy.stats
    np.random.seed(0)
    N = 5

    nfrX = sp.stats.binom.rvs(2, 0.3, size=(N, 10))
    nbgX = sp.stats.binom.rvs(2, 0.5, size=(N, 10))
    y = np.random.randint(0, 2, size=N)
    # r = apply_gcta(nfrX, nbgX, y, 0.5)

    K = grm_unbiased(nbgX)
    print(np.diagonal(K))
    # diag 0.8817461 0.9085317 0.6531746 1.2656746 0.5007936
    # print np.mean([0.8817461, 0.9085317, 0.6531746, 1.2656746, 0.5007936])

    #  [1] -0.06626985  0.15158729 -0.44087306 -0.62341273 -0.30337301 -0.32182536
    #  [7] -0.34365076 -0.09801586 -0.04206349 -0.01706349
