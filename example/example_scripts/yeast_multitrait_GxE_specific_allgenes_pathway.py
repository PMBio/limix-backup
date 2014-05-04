import limix.modules.qtl as QTL
import scipy as SP
import pandas as pd

#genes from lysine biosynthesis pathway
lysine_group = ['YIL094C', 'YDL182W', 'YDL131W', 'YER052C', 'YBR115C', 'YDR158W',
                'YNR050C', 'YJR139C', 'YIR034C', 'YGL202W', 'YDR234W']

pvalues_dict={}

for gene_idx,gene in enumerate(lysine_group):
    # select all environmens for gene YJR139C in lysine_group
    phenotype_query = "(gene_ID == '%s')" % gene
    print "running GxE GWAS tests on %s" % phenotype_query
    data_subsample = data.subsample_phenotypes(phenotype_query=phenotype_query,intersection=True)

    #get variables we need from data
    phenotypes,sample_idx = data_subsample.getPhenotypes(phenotype_query=phenotype_query,intersection=True); assert sample_idx.all()
    if phenotypes.shape[1]==0:
        continue
    
    snps = data_subsample.getGenotypes(impute_missing=True)
    
    K = data_subsample.getCovariance()
    pos = data_subsample.getPos()

    #set parameters for the analysis
    N, P = phenotypes.shape          

    covs = None                 #covariates
    Acovs = None                #the design matrix for the covariates   
    Asnps0 = SP.ones((1,P))     #the null model design matrix for the SNPs
    Asnps1 = SP.zeros((2,P))    #the alternative model design matrix for the SNPs
    Asnps1[0,:] = 1.0           
    Asnps1[1,0] = 1.0       
    K1r = K                     #the first sample-sample covariance matrix (non-noise)
    K2r = SP.eye(N)             #the second sample-sample covariance matrix (noise)
    K1c = None                  #the first phenotype-phenotype covariance matrix (non-noise)
    K2c = None                  #the second phenotype-phenotype covariance matrix (noise)
    covar_type = 'freeform'     #the type of covariance matrix to be estimated for unspecified covariances 
    searchDelta = False         #specify if delta should be optimized for each SNP
    test="lrt"                  #specify type of statistical test

    # Running the analysis
    # when cov are not set (None), LIMIX considers an intercept (covs=SP.ones((N,1)))
    pvalues = QTL.test_interaction_lmm_kronecker(snps=snps,phenos=phenotypes.values,covs=covs,Acovs=Acovs,Asnps1=Asnps1,Asnps0=Asnps0,K1r=K1r,K2r=K2r,K1c=K1c,K2c=K2c,trait_covar_type=covar_type,searchDelta=searchDelta)

    #convert P-values to a DataFrame for nice output writing:
    pvalues = pd.DataFrame(data=SP.concatenate(pvalues).T,index=data_subsample.geno_ID,columns=["specific","null_common","alternative_any"])
    pvalues = pd.concat([pos,pvalues],join="outer",axis=1)
    key = "pvalues_%s" % gene
    pvalues_dict[key] = pvalues
#create result DataFrame
result["pvalues"] = pvalues_dict