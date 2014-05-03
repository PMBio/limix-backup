import limix.modules.qtl as QTL
import scipy as SP
import pandas as pd

phenotype_query = "(gene_ID=='YBR115C') & (environment==0)"
#phenotype_query = "(environment==0)"

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

#run the analysis
#pvalues = QTL.simple_interaction_kronecker(snps=snps,phenos=phenotypes,covs=covs,Acovs=Acovs,Asnps1=Asnps1,Asnps0=Asnps0,K1r=K1r,K2r=K2r,K1c=K1c,K2c=K2c,covar_type=covar_type,rank=rank,searchDelta=searchDelta)

lmm = QTL.test_lmm(snps=snps,pheno=phenotypes.values,K=K,covs=covs, test=test)
pvalues = lmm.getPv()
#pvalues = SP.randn(phenotypes.shape[1],data.num_snps)
pvalues = pd.DataFrame(data=pvalues.T,index=data_subsample.geno_ID,columns=phenotypes.columns)
pvalues = pd.concat([pos,pvalues],join="outer",axis=1)
#create result DataFrame
result["pvalues"] = pvalues
    