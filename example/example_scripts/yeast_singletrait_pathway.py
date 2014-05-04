import limix.modules.qtl as QTL
import scipy as SP
import pandas as pd

#genes from lysine biosynthesis pathway
lysine_group = ['YIL094C', 'YDL182W', 'YDL131W', 'YER052C', 'YBR115C', 'YDR158W',
                'YNR050C', 'YJR139C', 'YIR034C', 'YGL202W', 'YDR234W']

#create a complex query on the gene_ID and environment:
# select environment 0 for all genes in lysine_group
phenotype_query = "(gene_ID in %s) & (environment==0)" % str(lysine_group)

data_subsample = data.subsample_phenotypes(phenotype_query=phenotype_query,intersection=True)

#get variables we need from data
snps = data_subsample.getGenotypes(impute_missing=True)
phenotypes,sample_idx = data_subsample.getPhenotypes(phenotype_query=phenotype_query,intersection=True); assert sample_idx.all()

K = data_subsample.getCovariance()
pos = data_subsample.getPos()

#set parameters for the analysis
N, P = phenotypes.shape          

covs = None                 #covariates
Acovs = None                #the design matrix for the covariates   
Asnps1 = SP.eye(P)          #the alternative model design matrix for the SNPs
Asnps0 = SP.ones((1,P))     #the null model design matrix for the SNPs
K1r = K                     #the first sample-sample covariance matrix (non-noise)
K2r = SP.eye(N)             #the second sample-sample covariance matrix (noise)
K1c = None                  #the first phenotype-phenotype covariance matrix (non-noise)
K2c = None                  #the second phenotype-phenotype covariance matrix (noise)
covar_type = 'lowrank_diag' #the type of covariance matrix to be estimated for unspecified covariances 
rank = 1                    #the rank of covariance matrix to be estimated for unspecified covariances (in case of lowrank)
searchDelta = False         #specify if delta should be optimized for each SNP
test="lrt"                  #specify type of statistical test

# Running the analysis
# when cov are not set (None), LIMIX considers an intercept (covs=SP.ones((N,1)))
lmm = QTL.test_lmm(snps=snps,pheno=phenotypes.values,K=K,covs=covs, test=test)

pvalues = lmm.getPv()       # 1xS vector of p-values (S=X.shape[1])
#convert P-values to a DataFrame for nice output writing:
pvalues = pd.DataFrame(data=pvalues.T,index=data_subsample.geno_ID,columns=phenotypes.columns)
pvalues = pd.concat([pos,pvalues],join="outer",axis=1)

betas = lmm.getBetaSNP()    # 1xS vector of effect sizes (S=X.shape[1])
#convert betas to a DataFrame for nice output writing:
betas = pd.DataFrame(data=betas.T,index=data_subsample.geno_ID,columns=phenotypes.columns)
betas = pd.concat([pos,pvalues],join="outer",axis=1)

#create result DataFrame
result["pvalues"] = pvalues
result["betas"] = betas
    