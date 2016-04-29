import scipy as SP
from . import par_lmm_forest as parUtils
import random

def checkMaf(X, maf=None):
    if maf==None:
        maf = 1.0/X.shape[0]
    Xmaf = (X>0).sum(axis=0)
    Iok = (Xmaf>=(maf*X.shape[0]))
    return SP.where(Iok)[0]

def scale_K(K, verbose=False):
    """scale covariance K such that it explains unit variance"""
    c = SP.sum((SP.eye(len(K)) - (1.0 / len(K)) * SP.ones(K.shape)) * SP.array(K))
    scalar = (len(K) - 1) / c
    if verbose:
        print(('Kinship scaled by: %0.4f' % scalar))
    K = scalar * K
    return K

def update_Kernel(unscaled_kernel, X_upd_in, scale=True):
    #filter and scale SNPs
    X_upd = X_upd_in.copy()
    X_upd = X_upd[:,checkMaf(X_upd)]
    X_upd -= X_upd.mean(axis=0)
    X_upd /= X_upd.std(axis=0)
    X_upd = X_upd.T
    #update kernel
    kernel_out = unscaled_kernel.copy()
    kernel_out -= SP.dot(X_upd.T, X_upd)
    if scale:
        return scale_K(kernel_out)
    else:
        return kernel_out

def estimateKernel(X, msample=None, maf=None, scale=True):
    #1. maf filter
    Xpop = X.copy()
    Xpop = Xpop[:,checkMaf(X, maf)]
    #2. sampling of predictors
    if msample != None:
        msample = SP.random.permutation(X.shape[1])[:msample]
        Xpop = Xpop[:,msample]
    Xpop -= Xpop.mean(axis=0)
    Xpop /= Xpop.std(axis=0)
    Xpop = Xpop.copy().T
    Xpop = SP.array(Xpop, dtype='float')
    Kpop  =  SP.dot(Xpop.T,Xpop)
    if scale:
        return scale_K(Kpop)
    else:
        return Kpop

def k_fold_cross_validation(items, k, randomize=True, seed=True):
    if randomize:
        if seed:
            random.seed(10) # make shure we get similar partitions across methods
        items = list(items)
        random.shuffle(items)
    slices = [items[i::k] for i in range(k)]

    for i in range(k):
        validation = slices[i]
        training = [item
                    for s in slices if s is not validation
                    for item in s]
        yield validation

def crossValidationScheme(folds, n):
    validationList = []
    for validation in k_fold_cross_validation(list(range(n)), folds):
        indexes = SP.ones(n) == 0
        indexes[validation] = True
        validationList.append(indexes)
    return validationList

def crossValidate(y, X, K=None, folds=3, model=None, returnModel=False):
    errors =  SP.empty(folds)
    n = y.shape[0]
    indexes = crossValidationScheme(folds,n)
    predictions = SP.empty(y.shape)
    alpha = []
    alphas = []
    msePath = []
    for cvRun in SP.arange(len(indexes)):
        testIndexes = indexes[cvRun]
        yTrain = y[~testIndexes]
        XTrain = X[~testIndexes]
        if K == None:
            model.fit(XTrain, yTrain)
            prediction = SP.reshape(model.predict(X[testIndexes]), (-1,1))
        else: # models having population structure
            KTrain = K[~testIndexes]
            KTrain = KTrain[:,~testIndexes]
            KTest=K[testIndexes]
            KTest=KTest[:,~testIndexes]
            model.reset()
            model.kernel = KTrain #TODO: make nice integration
            model.fit(XTrain, yTrain)
            prediction = SP.reshape(model.predict(X[testIndexes], k=KTest), (-1,1))
        predictions[testIndexes] = prediction
        errors[cvRun] = predictionError(y[testIndexes], prediction)
        print(('prediction error right now is', errors[cvRun]))
        if returnModel:
            alpha.append(model.alpha)
            alphas.append(model.alphas)
            msePath.append(model.mse_path)
    if returnModel:
        return indexes, predictions, errors, alpha, alphas, msePath
    else:
        return indexes, predictions, errors

def predictionError(yTest, yPredict):
    return ((yTest - yPredict)**2).sum()/SP.float_(yTest.shape[0])

def getQuadraticKernel(X, d=.01):
    K = SP.empty((X.shape[0], X.shape[0]))
    for i in SP.arange(X.shape[0]):
        for j in  SP.arange(X.shape[0]):
            K[i,j] = SP.exp(-0.5/d*(X[i]-X[j])**2)
    return scale_K(K)

def generate_linear_data(n_max, n_step, ssv_g, var):
    x = SP.arange(0,n_max,n_step).reshape(-1,1)
    y = SP.zeros_like(x).reshape(-1,1)*0.0
    X = convertToBinaryPredictor(x)
    Xbg = (SP.random.rand(X.shape[0], X.shape[1]) < .5) * 1.0
    weights = var*SP.random.randn(2,1)
    y += X[:,3:4] * weights[0,:]
    Xbg[:,3:4] = X[:,3:4]
    l = X[:,1:2] * X[:,2:3]
    Xbg[:,1:2] = X[:,1:2]
    Xbg[:,2:3] = X[:,2:3]
    y += l * weights[1,:]
    yTr = y.copy()
    ssv_v = 1.0-ssv_g
    if ssv_g > 0.0:
        ldelta = SP.log(ssv_v/SP.float_(ssv_g))
        K = scale_K(getQuadraticKernel(x, d=20))
    else:
        ldelta = None
        K = SP.eye(y.shape[0])
    y += SP.random.multivariate_normal(SP.zeros(K.shape[0]),ssv_g*K+ssv_v*SP.eye(K.shape[0])).reshape(-1,1)
    return Xbg, x, y, yTr, K, ldelta

def convertToBinaryPredictor(x):
    arr = []
    a = 0
    for i in SP.arange(x.size):
        arr.append(bin(x[i,0])[2:])
        l = max(a, bin(x[i,0])[2:].__len__())
    X = SP.zeros((x.size,l))

    for i in SP.arange(x.size):
        head0=l-arr[i].__len__()
        for j in SP.arange(head0):
            X[i,j] = 0
        for j in SP.arange(arr[i].__len__()):
            X[i,head0+j] = SP.int16(arr[i][j])
    return X

# generates data sets to test the continous version of the mixed forest
def lin_data_cont_predictors(n=100, m=1):
    X = SP.random.randn(n,m)
    beta = SP.random.randn(m,1)
    beta[1:]=0
    y = SP.dot(X,beta)
    return X, y
