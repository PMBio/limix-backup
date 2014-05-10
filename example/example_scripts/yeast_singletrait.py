import limix.modules.qtl as QTL
import scipy as SP
import pandas as pd

#create a complex query on the gene_ID and environment:
# select environment 0 for gene YBR115C
phenotype_query = "(gene_ID=='YBR115C') & (environment==0)"

data_subsample = data.subsample_phenotypes(phenotype_query=phenotype_query,intersection=True)

#get variables we need from data
snps = data_subsample.getGenotypes(impute_missing=True)
phenotypes,sample_idx = data_subsample.getPhenotypes(phenotype_query=phenotype_query,intersection=True); assert sample_idx.all()

sample_relatedness = data_subsample.getCovariance()
pos = data_subsample.getPos()

#set parameters for the analysis
N, P = phenotypes.shape          

covs = None                 #covariates
searchDelta = False         #specify if delta should be optimized for each SNP
test="lrt"                  #specify type of statistical test

# Running the analysis
# when cov are not set (None), LIMIX considers an intercept (covs=SP.ones((N,1)))
lmm = QTL.test_lmm(snps=snps,pheno=phenotypes.values,K=sample_relatedness,covs=covs,test=test)

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