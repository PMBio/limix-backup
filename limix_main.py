#all scripts included here are examined by cxfreeze which simplifies sorting out the include list..
import scipy as SP
import scipy.sparse.csgraph._validation
import limix
#import md5
#from IPython import embed

print 'hello, running script'
script_name = 'limix_script.py'

interactive_shell=True

#faking data generation
S = 1000
N = 100
P = 1

#generate fake data
X = SP.randn(N,S)
K = SP.eye(N)
snps = SP.randn(N,S)
pheno = SP.randn(N,P)
covs  = SP.ones([N,1])

#execute script in global namespace
execfile(script_name,globals())
#launch interactive shaell?
#if interactive_shell:
#	embed()

#data storage stuff..
