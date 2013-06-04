"""LMM testing code"""
import scipy as SP
import pdb
import sys
import limix


def CLMM_test():
    """simple test of CLMM"""
    print "Running CLMM test"

    SP.random.seed(1)
    #simulate genotype
    N = 100
    S = 10
    
    X = SP.random.randn(N,S)
    Y = SP.dot(X,SP.random.randn(S,1))
    Y += SP.random.randn(N,1)
    C = SP.ones([N,1])
    K = SP.eye(N)

    lmm = limix.CLMM()
    lmm.setK(K)
    lmm.setSNPs(X)
    lmm.setCovs(C)
    lmm.setPheno(Y)
    lmm.process()
    pv = lmm.getPv().ravel()
    D2= (pv-SP.array([  2.51035504e-01,   7.87110691e-11,   4.97690005e-01,4.64168307e-01,   1.35978541e-03,   5.78090595e-01, 5.79741538e-02,   4.27432447e-04,   3.35524125e-01,3.94149107e-06]))**2
    RV = SP.sqrt(D2.mean())<1E-6
    if RV:
        print "passed"
    else:
        print "failed"
    return RV


def CILMM_test():
    """Interaction test"""
    print "fill me"
    return True


def run_all_tests():
    RV = True
    RV = RV & CLMM_test()
    RV = RV & CILMM_test()
    return RV


if __name__ == '__main__':
    RV= run_all_tests()
    if RV:
        sys.exit(0)
    else:
        sys.exit(99)
