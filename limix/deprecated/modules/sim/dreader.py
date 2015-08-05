import h5py
import numpy as np

class DReader(object):
    def choose_parents(self, pop_id, nparents):
        raise NotImplemented

    def generate_genotype(self, par_indices, chroms):
        raise NotImplemented

    def mean_std(self, chrom, pop_ids):
        raise NotImplemented

    @property
    def pop_ids(self):
        raise NotImplemented

    def pop_size(self, pop_id):
        raise NotImplemented

    @property
    def nindividuals(self):
        raise NotImplemented

def _random_sep_points(nseps, arr_size):
    sep = np.zeros(arr_size + nseps)
    sep[0:nseps] = 1.
    np.random.shuffle(sep)
    sep_points = np.where(sep == 1.0)[0]
    return sep_points

def _cross_parents(par_genos, sep_points):
    geno = list(par_genos[0, 0:sep_points[0]])
    for i in xrange(1, len(sep_points)):
        a = sep_points[i-1] - (i-1)
        b = sep_points[i] - i
        geno.extend(list(par_genos[i, a:b]))

    a = sep_points[-1] - len(sep_points) + 1
    geno.extend(list(par_genos[-1, a:]))
    return np.array(geno)

def normalize_genotype(genotype, mean_std):
    return (genotype - mean_std[0]) / mean_std[1]

class DReader1000G(object):
    pop2subpop = dict(EUR=('FIN', 'GBR', 'IBS', 'CEU', 'TSI'),
                      EAS=('CHS', 'CHB', 'JPT'),
                      AFR=('ASW', 'LWK', 'YRI'),
                      AMR=('MXL', 'CLM', 'PUR'))

    def __init__(self, filepath, maf=0.05, pop_ids='all'):
        self._filepath = filepath
        self._pop_ids = pop_ids
        self._maf = maf

        self._chrom_sizes = dict()
        self._valid_snps_cache = dict()
        self._file = h5py.File(self._filepath, 'r')

        inds = self._file['genotypes']['chrom22']['row_headers']
        self._ind2subpop = inds['population'][:]
        for (k, v) in self._file['genotypes'].iteritems():
            if 'chrom' in k:
                siz = v['matrix_inds_by_snps_c'].shape[1]
                self._chrom_sizes[k] = siz

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self._file.close()

    def close(self):
        self._file.close()

    def nsnps(self, chroms):
        s = 0
        for k in chroms:
            valid_snps = self._valid_snps(k)
            s += len(valid_snps)
        return s

    def pop_size(self, pop_id):
        if pop_id == 'all':
            s = 0
            for p in self.pop_ids:
                s += self.pop_size(p)
            return s
        return len(self._individuals_from_pop(pop_id))

    def choose_parents(self, nparents=10):
        inds = self._individuals_from_pops()

        if nparents >= len(inds):
            par_indices = ids
        else:
            par_indices = np.random.choice(inds, size=nparents, replace=False)
        return par_indices

    def _parent_genotypes(self, par_indices, chroms):
        par_genos = []
        for pi in par_indices:
            ogeno = self._original_genotype(pi, chroms)
            par_genos.append(ogeno)
        return np.array(par_genos)

    def generate_genotype(self, par_indices, chroms):
        nparents = len(par_indices)

        par_genos = self._parent_genotypes(par_indices, chroms)
        nsnps = len(par_genos[0])
        sep_points = _random_sep_points(nparents - 1, nsnps)

        geno = _cross_parents(par_genos, sep_points)

        return geno

    def mean_std(self, chroms):
        pop_ids = self._pop_ids
        if pop_ids == 'all':
            pop_ids = DReader1000G.pop2subpop.keys()

        individuals = self._individuals_from_pops()

        genotypes = []

        genos = []
        for k in chroms:
            f0 = self._file['genotypes']['chrom%d' % k]
            valid_snps = self._valid_snps(k)
            gs = []
            for ind in individuals:
                g = f0['matrix_inds_by_snps_c'][ind, :]
                gs.append(g[valid_snps])
            genos.append(np.asarray(gs))

        genos = np.hstack(genos)
        m = np.mean(genos, axis=0)
        v = np.std(genos, axis=0)

        return (m, v)

    def _original_genotype(self, individual, chroms):
        pop_ids = self._pop_ids
        maf = self._maf
        geno = []
        for chrom in chroms:
            f0 = self._file['genotypes']['chrom%d' % chrom]
            valid_snps = self._valid_snps(chrom)
            g = f0['matrix_inds_by_snps_c'][individual, :]
            g = g[valid_snps]
            geno.extend(list(g))
        return geno

    def _individuals_from_pops(self):
        pop_ids = self._pop_ids
        individuals = []
        for pop_id in pop_ids:
            inds = self._individuals_from_pop(pop_id)
            individuals.extend(inds)
        individuals = np.sort(individuals)
        return individuals

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

    def _valid_snps(self, chrom):
        maf = self._maf
        pop_ids = self._pop_ids

        key = str((chrom, pop_ids, maf))
        if key in self._valid_snps_cache:
            return self._valid_snps_cache[key]
        individuals = self._individuals_from_pops()
        f0 = self._file['genotypes']['chrom%d' % chrom]
        genos = (f0['matrix_inds_by_snps_c'][individuals, :])
        us = np.unique(genos)
        min_count = np.full(genos.shape[1], np.inf)

        for u in us:
            s = np.sum(genos == u, 0)
            min_count = np.minimum(min_count, s)
        mafs = min_count / float(genos.shape[0])
        self._valid_snps_cache[key] = mafs >= maf
        return np.where(self._valid_snps_cache[key])[0]

if __name__ == '__main__':
    dr = DReader1000G('/Users/horta/workspace/1000G_majors.hdf5')
    parents = dr.choose_parents('EUR', 3)
    ind0 = dr.generate_genotype(parents, [22, 1])
    (m, s) = dr.mean_std(22, 'all')
    print dr.pop_size('all')
    print dr.pop_ids
