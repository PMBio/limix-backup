'''
Created on Feb 20, 2013

@author: johannes
'''

import scipy.linalg as LA
import scipy as SP
from . import lmm_fast
import limix

class BLUP(object):
    '''
    classdocs
    '''
    def __init__(self):
        '''
        Constructor
        '''
    def reset(self):
        self = BLUP()

    def fit(self, XTrain=None, yTrain=None, KTrain=None, delta=None, folds=0):
        self.nTrain = yTrain.shape[0]
        self.yTrain = yTrain
        if KTrain is not None:
            self.kernel = KTrain
        self.S, self.U = LA.eigh(self.kernel+SP.eye(self.nTrain)*1e-8)
        self.Uy = SP.dot(self.U.T, yTrain)
        self.Uone = SP.dot(self.U.T, SP.ones_like(yTrain))
        if delta is None:
            # Initialize cross-learning of delta
            if folds > 0:
                self.ldelta = self.crossvalidate_delta(folds)
            else:
                self.ldelta = lmm_fast.optdelta(self.Uy[:, 0], self.Uone, self.S,
                                                numintervals=1000)
            self.delta = SP.exp(self.ldelta)
        else:
            self.delta = delta
            self.ldelta = SP.log(delta)
        self.cov = LA.inv((self.kernel + SP.eye(self.nTrain)*self.delta))

    def update_delta(self, mean):
        Umean = SP.dot(self.U.T, mean).reshape(-1, 1)
        self.ldelta = lmm_fast.optdelta(self.Uy[:, 0], Umean, self.S, ldeltamin=self.ldelta-1., ldeltamax=self.ldelta+1.)
        self.delta = SP.exp(self.ldelta)
        return self.delta

    def crossvalidate_delta(self, folds):
        import utils
        cv_scheme = utils.crossValidationScheme(folds, self.nTrain)
        ldeltas = SP.arange(-3, -1.5, .01)
        Ss = []
        Us = []
        Uys = []
        UCs = []
        err = 0.0
        errs = []
        for ldelta in ldeltas:
            for test_set in cv_scheme:
                train_set = ~test_set
                K_sub = self.kernel[SP.ix_(train_set, train_set)]
                K_cross = self.kernel[SP.ix_(~train_set, train_set)]
                # print LA.inv((K_sub + SP.eye(train_set.sum())*self.delta))
                Core = SP.dot(K_cross, LA.inv((K_sub + SP.eye(train_set.sum()) *
                                               SP.exp(ldelta))))
                diff = self.yTrain[test_set] -\
                    SP.dot(Core, self.yTrain[train_set])
                err += (diff**2).sum()/diff.size
                S, U = LA.eigh(self.kernel[SP.ix_(train_set, train_set)])
                Ss.append(S)
                Us.append(U)
                Uys.append(SP.dot(U.T, self.yTrain[train_set]))
                UCs.append(SP.dot(U.T, SP.ones_like(self.yTrain[train_set])))
            errs.append(err/len(cv_scheme))
            err = 0.0

        nll_scores = []
        for ldelta in ldeltas:
            # print 'ldelta equals', ldelta
            score = 0.0
            for i in range(len(cv_scheme)):
                score += lmm_fast.nLLeval(ldelta, (Uys[i])[:, 0], UCs[i], Ss[i])
            nll_scores.append(score/len(cv_scheme))
        print(('best ldelta found ll', ldeltas[SP.argmin(nll_scores)]))
        return ldeltas[SP.argmin(errs)]

    def LL(self, mean):
        mean.reshape(-1, 1)
        res = self.yTrain - mean
        ll = SP.dot(SP.dot(res.T, self.cov), res)
        return ll

    def predict(self, XTest=None, k=None, mean=None):
        # k is cross covariance KTestTrain
        self.mean = mean
        if self.mean is None:
            self.mean = SP.ones_like(self.yTrain)*self.yTrain.mean()
        if k is None:
            k = self.kernel
        Core = SP.dot(k, LA.inv((self.kernel + SP.eye(self.nTrain)*self.delta)))
        return SP.dot(Core, self.yTrain-self.mean)
