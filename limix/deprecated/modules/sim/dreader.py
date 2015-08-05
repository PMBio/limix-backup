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

def normalize_genotype(genotype, mean_std):
    return (genotype - mean_std[0]) / mean_std[1]

class DReader1000G(object):
    pop2subpop = dict(EUR=('FIN', 'GBR', 'IBS', 'CEU', 'TSI'),
                      EAS=('CHS', 'CHB', 'JPT'),
                      AFR=('ASW', 'LWK', 'YRI'),
                      AMR=('MXL', 'CLM', 'PUR'))

    def __init__(self, filepath):
        self._filepath = filepath

        self._chrom_sizes = dict()
        self._file = h5py.File(self._filepath, 'r')

        inds = self._file['genotypes']['chrom22']['row_headers']
        self._ind2subpop = inds['population'][:]
        for (k, v) in self._file['genotypes'].iteritems():
            if 'chrom' in k:
                siz = v['matrix_inds_by_snps'].shape[1]
                self._chrom_sizes[k] = siz

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self._file.close()

    def close(self):
        self._file.close()

    @property
    def pop_ids(self):
        return DReader1000G.pop2subpop.keys()

    def pop_size(self, pop_id):
        if pop_id == 'all':
            s = 0
            for p in self.pop_ids:
                s += self.pop_size(p)
            return s
        return len(self._individuals_from_pop(pop_id))

    def choose_parents(self, pop_id, nparents=10):
        inds = self._individuals_from_pop(pop_id)
        if nparents >= len(inds):
            par_indices = ids
        else:
            par_indices = np.random.choice(inds, size=nparents, replace=False)
        return par_indices

    def generate_genotype(self, par_indices, snpsref):
        nparents = len(par_indices)

        par_genos = []
        for pi in par_indices:
            ogeno = self._original_genotype(pi, snpsref)
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

    def mean_std(self, snpsref, pop_ids='all'):
        if pop_ids == 'all':
            pop_ids = DReader1000G.pop2subpop.keys()

        individuals = []
        for pop_id in pop_ids:
            inds = self._individuals_from_pop(pop_id)
            individuals.extend(inds)
        individuals = np.sort(individuals)

        genotypes = []

        import time
        start = time.time()
        for ind in individuals:
            genos = np.empty(0)
            for (k, v) in snpsref.iteritems():
                f0 = self._file['genotypes']['chrom%d' % k]
                genos = np.append(genos, f0['matrix_inds_by_snps_c'][ind, v])
            genotypes.append(genos)
        print "Time %.5fs" % (time.time()-start)
        genotypes = np.asarray(genotypes, float)
        m = np.mean(genos, axis=0)
        v = np.std(genos, axis=0)

        return (m, v)

    def snps_reference(self, chroms, nsnps):
        tsize = 0
        for chrom in chroms:
            tsize += self._chrom_sizes['chrom%d' % chrom]

        if nsnps == 'all' or nsnps >= tsize:
            nsnps = tsize
        idx = np.asarray(np.linspace(0, tsize, nsnps, endpoint=False),
                         int)
        subidx = dict()
        for chrom in chroms:
            s = self._chrom_sizes['chrom%d' % chrom]
            subidx[chrom] = idx[:np.sum(idx < s)]
            idx = idx[len(subidx[chrom]):]
            if len(idx) > 0:
                idx -= idx[0]
        assert len(idx) == 0, ("There shouldn't be remaining SNPs "
                               "to be selected.")
        return subidx

    def _original_genotype(self, individual, snpsref):
        geno = []
        for (chrom, snps) in snpsref.iteritems():
            f0 = self._file['genotypes']['chrom%d' % chrom]
            g = f0['matrix_inds_by_snps'][individual, snps]
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
    par_parents = dr.choose_parents('EUR', 3)
    ind0 = dr.generate_genotype(par_parents, [22, 1])
    (m, s) = dr.mean_std(22, 'all')
    print dr.pop_size('all')
    print dr.pop_ids
