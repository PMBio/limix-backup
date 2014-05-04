import limix.modules.varianceDecomposition as VAR
import scipy as SP
import pandas as pd

#genes from lysine biosynthesis pathway
lysine_group = ['YIL094C', 'YDL182W', 'YDL131W', 'YER052C', 'YBR115C', 'YDR158W',
                'YNR050C', 'YJR139C', 'YIR034C', 'YGL202W', 'YDR234W']

var = []
genes_analyzed = []
K_all = data.getCovariance(normalize=False)
    
for gene_idx,gene in enumerate(lysine_group):
    #create a complex query on the gene_ID and environment:
    # select environment 0 for all genes in lysine_group
    phenotype_query = "(gene_ID=='%s') & (environment==0)" % gene
    print "running variance decomposition on %s" % phenotype_query
    data_subsample = data.subsample_phenotypes(phenotype_query=phenotype_query,intersection=True)
    
    #get variables we need from data
    phenotypes,sample_idx = data_subsample.getPhenotypes(phenotype_query=phenotype_query,intersection=True); assert sample_idx.all()
    if phenotypes.shape[1]==0:
        continue
    #snps = data_subsample.getGenotypes(impute_missing=True)
    pos_gene = data_subsample.pheno_reader.get_pos(phenotype_query=phenotype_query)
    K_cis = data_subsample.getCovariance(normalize=False,pos_start=pos_gene["start"][0],pos_end=pos_gene["end"][0],windowsize=5e5)
    K_trans = K_all-K_cis
    K_cis /= K_cis.diagonal().mean()
    K_trans /= K_trans.diagonal().mean()
    
    # variance component model
    vc = VAR.VarianceDecomposition(phenotypes.values)
    vc.addFixedEffect()
    vc.addRandomEffect(K=K_cis)
    vc.addRandomEffect(K=K_trans)
    vc.addRandomEffect(is_noise=True)
    vc.optimize()
    
    # get variances
    # vc.getVariances() returs a vector of variances explained
    # by the three random effects in order of addition (cis,trans,noise)
    _var = vc.getVarianceComps()
    
    var.append(_var)
    genes_analyzed.append(gene)
# concatenate in a unique matrix
var = SP.concatenate(var)
#normalize variance component and average
var/=var.sum(1)[:,SP.newaxis]
var_mean = var.mean(0)[SP.newaxis,:]

column_labels = ["cis","trans","noise"]

#convert var to a DataFrame for nice output writing:
var = pd.DataFrame(data=var,index=genes_analyzed,columns=column_labels)

#convert betas to a DataFrame for nice output writing:
var_mean = pd.DataFrame(data=var_mean,index=["lysine_biosynthesis"],columns=column_labels)

#create result DataFrame
result["variance_components_gene"] = var
result["variance_components_mean"] = var_mean
    