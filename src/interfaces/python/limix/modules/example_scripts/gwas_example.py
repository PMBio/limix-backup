import limix.modules.qtl as QTL
import scipy as SP

pheno_names_complete = SP.array(['Calcium_Chloride', 'Cisplatin', 'Cobalt_Chloride', 'Congo_red',
       'Copper', 'Cycloheximide', 'Diamide', 'E6_Berbamine', 'Ethanol',
       'Formamide', 'Galactose', 'Hydrogen_Peroxide', 'Hydroxyurea',
       'Indoleacetic_Acid', 'Lactate', 'Lactose', 'Lithium_Chloride',
       'Magnesium_Chloride', 'Magnesium_Sulfate', 'Maltose', 'Mannose',
       'Menadione', 'Neomycin', 'Paraquat', 'Raffinose', 'Tunicamycin',
       'x4-Hydroxybenzaldehyde', 'x4NQO', 'x5-Fluorocytosine',
       'x5-Fluorouracil', 'x6-Azauracil', 'Xylose', 'YNB', 'YNB:ph3',
       'YNB:ph8', 'YPD:15C', 'YPD:37C', 'YPD:4C'])


#get variables we need from data
X = data.getGenotypes()
K = data.getCovariance()
pos = data.getPos()

[Y,Ikeep] = data.getPhenotypes(phenotype_IDs=pheno_names_complete,intersection=True)
X = X[Ikeep].copy()
K = K[Ikeep,:][:,Ikeep].copy()

#set parameters for the analysis
phenos=Y                    #phenotypes
covs=None                   #covariates
Acovs=None                  #the design matrix for the covariates   
Asnps1=SP.eye(P)            #the alternative model design matrix for the SNPs
Asnps0=SP.ones((1,P))       #the null model design matrix for the SNPs
K1r=K                       #the first sample-sample covariance matrix (non-noise)
K2r=SP.eye(N)               #the second sample-sample covariance matrix (noise)
K1c=None                    #the first phenotype-phenotype covariance matrix (non-noise)
K2c=None                    #the second phenotype-phenotype covariance matrix (noise)
covar_type='lowrank_diag'   #the type of covariance matrix to be estimated for unspecified covariances 
rank=1                      #the rank of covariance matrix to be estimated for unspecified covariances (in case of lowrank)
searchDelta=False           #specify if delta should be optimized for each SNP

#run the analysis
result = {}     #create result dictionary
result['pvalues']=QTL.simple_interaction_kronecker(snps=snps,phenos=phenos,covs=covs,Acovs=Acovs,Asnps1=Asnps1,Asnps0=Asnps0,K1r=K1r,K2r=K2r,K1c=K1c,K2c=K2c,covar_type=covar_type,rank=rank,searchDelta=searchDelta)
result['a']=a

#import pandas as pd
#result['pvalues']=pd.DataFrame(data=pvalues.T,index=data.,columns=['pv', 'pv0', 'pvAlt'])