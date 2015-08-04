import h5py
import numpy as np

class DReader(object):
    @property
    def nindividuals(self):
        raise NotImplemented

    @property
    def npops(self):
        raise NotImplemented

    @property
    def nsnps(self):
        raise NotImplemented

def _random_sep_points(nseps, arr_size):
    sep = np.zeros(arr_size + nseps)
    sep[0:nseps] = 1.
    np.random.shuffle(sep)
    sep_points = np.where(sep == 1.0)[0]
    return sep_points

class DReader1000G(object):
    pop2subpop = dict(EUR=('FIN', 'GBR', 'IBS', 'CEU', 'TSI'),
                      EAS=('CHS', 'CHB', 'JPT'),
                      AFR=('ASW', 'LWK', 'YRI'),
                      AMR=('MXL', 'CLM', 'PUR'))

    def __init__(self, filepath):
        self._filepath = filepath

        with h5py.File(self._filepath, 'r') as f:
            inds = f['genotypes']['chrom22']['row_headers']
            self._ind2subpop = inds['population'][:]

    def random_choose_parents(self, pop_id, nparents=10):
        inds = self._individuals_from_pop(pop_id)
        if nparents >= len(inds):
            par_indices = ids
        else:
            par_indices = np.random.choice(inds, size=nparents, replace=False)
        return par_indices

    def generate_individual(self, par_indices, chroms=[22]):
        nparents = len(par_indices)

        par_genos = []
        for pi in par_indices:
            ogeno = self._original_genotype(pi, chroms)
            par_genos.append(ogeno)
        par_genos = np.array(par_genos)
        nsnps = len(par_genos[0])
        sep_points = _random_sep_points(nparents - 1, nsnps)

        geno = list(par_genos[0, 0:sep_points[0]])
        for i in xrange(1, len(sep_points)):
            a = sep_points[i-1] - (i-1)
            b = sep_points[i] - i
            geno.extend(list(par_genos[i, a:b]))

        a = sep_points[-1] - len(sep_points) + 1
        b = nsnps
        geno.extend(list(par_genos[-1, a:b]))
        return np.array(geno)

    def _original_genotype(self, individual, chroms):
        geno = []
        with h5py.File(self._filepath, 'r') as f:
            for chrom in chroms:
                f0 = f['genotypes']['chrom%d' % chrom]
                g = f0['matrix_inds_by_snps'][individual, :]
                geno.extend(list(g))
        return geno

    def _individuals_from_pop(self, pop_id):
        pop2subpop = DReader1000G.pop2subpop
        assert pop_id in pop2subpop
        ids = []
        for subpop in pop2subpop[pop_id]:
            v = self._individuals_from_subpop(subpop)
            ids.extend(list(v))
        return np.array(ids)

    def _individuals_from_subpop(self, subpop_id):
        pop2subpop = DReader1000G.pop2subpop
        ids = np.where(self._ind2subpop == subpop_id)[0]
        return ids

if __name__ == '__main__':
    dr = DReader1000G('/Users/horta/workspace/1000G_majors.hdf5')
    par_parents = dr.random_choose_parents('EUR', 3)
    ind0 = dr.generate_individual(par_parents, [22, 1])
    print ind0.shape
