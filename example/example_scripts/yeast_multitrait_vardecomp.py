import limix.modules.varianceDecomposition as VAR
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
phenotypes,sample_idx = data_subsample.getPhenotypes(phenotype_query=phenotype_query,intersection=True); assert sample_idx.all()

K = data_subsample.getCovariance()
pos = data_subsample.getPos()

#set parameters for the analysis
N, P = phenotypes.shape          

# variance component model
vc = VAR.VarianceDecomposition(phenotypes.values)
vc.addFixedEffect()
vc.addRandomEffect(K=K,trait_covar_type='lowrank_diag',rank=4)
vc.addRandomEffect(is_noise=True,trait_covar_type='lowrank_diag',rank=4)
vc.optimize()
# retrieve geno and noise covariance matrix
Cg = vc.getTraitCovar(0)
Cn = vc.getTraitCovar(1)

#convert P-values to a DataFrame for nice output writing:
genetic_covar = pd.DataFrame(data=Cg,index=phenotypes.columns,columns=phenotypes.columns)
noise_covar = pd.DataFrame(data=Cn,index=phenotypes.columns,columns=phenotypes.columns)


#create result DataFrame
result["genetic_pheno_covar"] = genetic_covar
result["noise_pheno_covar"] = noise_covar
    