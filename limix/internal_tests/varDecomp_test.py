import scipy as SP
import varianceDecomposition as VAR
import ipdb

SP.random.seed(0)

if __name__=='__main__':

    # generate data
    h2 = 0.3
    N = 1000; P = 4; S = 1000
    X = 1.*(SP.rand(N,S)<0.2)
    beta = SP.randn(S,P)
    Yg = SP.dot(X,beta); Yg*=SP.sqrt(h2/Yg.var(0).mean())
    Yn = SP.randn(N,P); Yn*=SP.sqrt((1-h2)/Yn.var(0).mean())
    Y  = Yg+Yn; Y-=Y.mean(0); Y/=Y.std(0)
    XX = SP.dot(X,X.T)
    XX/= XX.diagonal().mean()

    # add first fixed effect
    F1 = 1.*(SP.rand(N,2)<0.2); A1 = SP.eye(P)
    # add first fixed effect
    F2 = 1.*(SP.rand(N,3)<0.2); A2 = SP.ones((1,P))

    ipdb.set_trace()

    vc = VAR.VarianceDecomposition(Y)
    vc.addFixedEffect(F=F1,A=A1)
    vc.addFixedEffect(F=F2,A=A2)
    vc.addRandomEffect(XX,trait_covar_type='freeform')
    vc.addRandomEffect(is_noise=True,trait_covar_type='freeform')
    vc.optimize()

