import numpy as np

def _calculate_maf(G):
    assert len(G.shape) == 2
    u = np.unique(G[:])
    assert len(u) <= 3

    s = np.full(G.shape[1], np.inf)
    for ui in u:
        s = np.minimum(np.sum(G == ui, axis=0), s)

    print s / float(G.shape[0])

# According to GCTA's paper
def grm_unbiased(G):
    import ipdb; ipdb.set_trace()
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
    G = np.random.randint(0, 3, (10, 50))
    # _calculate_maf(G)
    K = grm_unbiased(G)
