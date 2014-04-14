import pdb
import limix.modules.qtl as QTL
#1. run simple LMM
lm=QTL.simple_lmm(snps,pheno,K,covs)
#2. get pva
pv = lm.getPv()
pass
