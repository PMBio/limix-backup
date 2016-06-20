from .gwas_multivariate import *

if __name__ == "__main__":

	bed_fn = "../../test/data/plinkdata/toydata"
	pheno_fn = bed_fn + ".phe6"#"../../test/data/plinkdata/toydata.phe"
	covariate_fn = 	bed_fn + ".phe"

	
	blocksize = 20000

	#create a Plink Bed reader object:
	snp_reader = Bed(bed_fn)#[:,0:50000]
	
	#unit variance and zero mean standardization:	
	standardizer = pysnptools.standardizer.Unit()

	#load the phenotype and covariates
	pheno = pysnptools.util.pheno.loadPhen(filename=pheno_fn,   missing ='-9')
	pheno = GWAS._pheno_fixup(pheno)
	covariates = GWAS._pheno_fixup(covariate_fn, iid_source_if_none=pheno)
	
	print("--------------------------------------\nintersecting data")
	t00 = time.time()
	snp_intersect, pheno_intersect, covariates_intersect = pysnptools.util.intersect_apply([snp_reader, pheno, covariates], sort_by_dataset=True)
	
	pheno_df = GWAS.check_pheno_format(pheno_intersect)
	covariates_df = GWAS.check_pheno_format(covariates_intersect)

	t1 = time.time()
	print("done intersecting after %.4fs\n--------------------------------------" % (t1-t00))

	print("building kernel")
	t0 = time.time()
	
	#building the kernel
	K = snp_intersect.kernel(standardizer=standardizer,blocksize=blocksize)
	K /= K.diagonal().mean()
	
	t1 = time.time()
	print("done building kernel after %.4fs\n--------------------------------------" % (t1-t0))	
	

	N = pheno_df.shape[0]
	P = pheno_df.shape[1]
	print("the data has %i samples, %i phenotypes, %i SNPs to test\n--------------------------------------" %(N,P,snp_reader.sid.shape[0]))
	Y = pheno_df.values
	
	print("fit the background GP model using GP2KronSum:")
	# define fixed effects
	F = []; A = []
	X_cov = np.concatenate((np.ones((N,1)),covariates_df.values),1)
	F.append(X_cov)
	A.append(np.eye(P))
	# define row covariance
	R  = K
	# define col covariances
	Cg = FreeFormCov(P)
	Cn = FreeFormCov(P)
	Cg.setCovariance(0.5 * np.eye(P))
	Cn.setCovariance(0.5 * np.cov(Y.T))
	# define gp
	gp = GP2KronSum(Y=Y, F=F, A=A, Cg=Cg, Cn=Cn, R=R)
	gp.optimize()

	#extract the optimal column covariance matrices
	C1 = gp.covar.Cg.K()
	C2 = gp.covar.Cn.K()

	#run GWAS using the background model
	print("--------------------------------------\nrunning GWAS")
	t0 = time.time()
	mygwas = MultivariateGWAS(K=K, snps_K=None, snps_test=snp_intersect, phenotype=pheno_df, covariates=covariates_df, h2=None, interact_with_snp=None, nGridH2=10, standardizer=standardizer,C1=C1,C2=C2)

	result = mygwas.compute_association(blocksize=blocksize, temp_dir=None)#'./temp_dir_testdata/')

	t1 = time.time()
	print("done running GWAS after %.4fs\n--------------------------------------" % (t1-t0))
	print("total: %.4fs\n--------------------------------------" % (t1-t00))

	print("this is how the first five rows of the result (total %i rows) look like:\n" % (result.shape[0]))
	pd.options.display.max_columns = 10
	print(result.head(5))


