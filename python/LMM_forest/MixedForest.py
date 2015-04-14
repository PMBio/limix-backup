'''
Created on Apr 2, 2013

@author: johannes
'''

import scipy as SP
import warnings
import pySplittingCore as SC
import MixedForestTree as mf_tree
import BLUP as BLUP
import parMixedForest
import mixedForestUtils as utils

class Forest(object):
    '''
    classdocs
    '''
    def __init__(self,
                 n_estimators=100,
                 max_depth=float('inf'),
                 min_samples_split=5,
                 ratio_features=.66,
                 compute_importances=False,
                 fit_optimal_depth=False,
                 build_to_opt_depth=False,
                 pruning='err',
                 min_depth=0,
                 update_delta=False,
                 optimize_memory_use = True,
                 oob_score=False,
                 tc=None,
                 random_state=None,
                 kernel='data',
                 delta=None,
                 subsampling=True,
                 sampsize=.5,
                 verbose=0,
                 max_nodes=float('inf'),
                 cpp_fit=True, 
                 cpp_predict=True):
        ''' 
        A linear mixed forest.
        A linear mixed forest is a meta estimator derived from the random forest
        that fits a number of linear mixed decision trees on various sub-samples of the data set and uses averaging
        to improve the predictive accuracy and control over-fitting.
    
        Parameters
        ----------
         
        kernel: accepts either
            (1) positive definite matrix of size n x n, describing a dependency structure between the n samples (e.g. population structure).
            (2) strings 'data' or 'iid'
            If set to 'data' dependency structure is estimated from the predictor matrix X, if set to 'iid' i.i.d. samples are assumed and the linear mixed forest behaves like the random forest.
           
        n_estimators : integer, optional (default=100)
            The number of trees in the forest.
    
        ratio_features : fraction of the total number of features used for each split (default=0.66)
    
        max_depth : optional (default='inf') real value that defining the maximum depth of the tree.
    
        min_samples_split : integer, optional (default=5)
            The minimum number of samples required to split a node.
    
        fit_opt_depth: uses the forest likelihood to fit the optimal depth of the trees in the ensemble
            on the out of bag samples (default=False)
            
        build_to_opt_depth: stops furthering trees in the ensemble if improvement of forest likelihood stops (default=False). This helps to improve runtime
           and to avoid overfitting when a relatively small number of trees is used on a big dataset.
        
        sampsize: The number of samples drawn without replacement for growing a single tree (default=0.5)
        
        subsampling: If 'True' this toggles sampling without replacement. Otherwise as sampling without replacement is used (default=True)
         
        verbose : int, optional (default=0)
            Controls the verbosity of the tree building process.
        '''
        self.init_parameters = locals()
        self.init_parameters.pop('self')
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.kernel = kernel
        self.delta = delta
        self.update_delta = update_delta
        self.compute_importances = compute_importances
        self.fit_optimal_depth = fit_optimal_depth
        self.build_to_opt_depth = build_to_opt_depth
        self.optimize_memory_use = optimize_memory_use
        self.oob_error = None
        self.var_used = None
        self.log_importance = None
        self.ratio_features = ratio_features
        self.subsampling = subsampling
        self.sampsize = sampsize
        self.verbose = verbose
        self.depth = 0
        self.opt_depth = None
        self.pruning = pruning
        self.min_oob_error = float('inf')
        self.opt_delta = None
        self.fixed_effect = None
        self.tc = tc
        self.max_nodes = max_nodes
        self.cpp_fit = cpp_fit
        self.cpp_predict = cpp_predict
        self.min_depth = min_depth
        
        ###############################
        #settings for (re)initialization#
        ###############################
        
        self.X = None
        self.y = None
        self.n = None 
        self.m = None
        self.BLUP = None
        self.max_features = None
        self.var_used = None
        self.log_importance = None
        self.trees = []
        
        if random_state != None:
            SP.random.seed(random_state)
        if tc != None:
            parMixedForest.par_init(**self.init_parameters)
            
    
    def reset(self):
        # reinstatiate using the same parameters
        self.__init__(**self.init_parameters)
        
    def get_params(self, deep=True):
        return self.init_parameters
    
    # TODO: this is quite a hack for making scikits crossvalidation work
    def set_params(self, params):
        self.__init__(**params)
        
    def fit(self, X, y, recycle=True, **grow_params):
        """Build a linear mixed forest of trees from the training set (X, y).

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The training input samples.

        y : array-like, shape = [n_samples] or [n_samples, 1]
            The real valued targets

        Returns
        -------
        self : object
            Returns self.
        """
        if self.kernel == 'data':
            self.kernel = SC.estimateKernel(X, maf=1.0/X.shape[0])
        elif self.kernel == 'iid':
            self.kernel = SP.identity(X.shape[0])
        elif self.kernel.size == X.shape[1]: #use dedicated part of data as background model
            tmp_ind = self.kernel
            self.kernel = utils.estimateKernel(X[:,self.kernel], maf=1.0/X.shape[0])
            X = X[:,~tmp_ind]
        # extract and reshape data
        self.y = y.reshape(-1,1)
        self.X = X
        self.n, self.m = self.X.shape
        if self.delta==None:
            self.BLUP = BLUP.BLUP()
            self.BLUP.fit(XTrain=self.X, yTrain=self.y, KTrain=self.kernel, delta=self.delta)
            self.delta = self.BLUP.delta #update delta if it used to be 'None'
        self.max_features = SP.maximum(SP.int_(self.ratio_features*self.m),1)
        self.var_used = SP.zeros(self.m)
        self.log_importance = SP.zeros(self.m)
        self.depth = 0
        
        if self.verbose > 0: 
            print 'log(delta) fitted to ', SP.log(self.delta)
            
        #####################################################   
        #Initialize individual trees
        if recycle and self.trees != []:
            for tree in self.trees:
                tree.cut_to_stump()
        else:
            n_trees = 0
            self.trees = []
            while n_trees < self.n_estimators:
                if self.verbose > 1:
                    print 'init. tree number ', n_trees
                subsample = self.tree_sample()
                tree = mf_tree.MixedForestTree(self, subsample)
                self.trees.append(tree) 
                n_trees += 1
        
        #####################################################
        #Fitting with optimal depth constraint
        if self.fit_optimal_depth or self.update_delta: #i.e. we need to fit all trees stepwise
            self.opt_depth = 0
            self.min_oob_err = self.get_oob_error(self.depth)
            if self.verbose > 0:
                print 'initial oob error is:', self.min_oob_err
            grow_further = True
            curr_depth = self.depth
            while grow_further:
                # updating ensemble increasing its depth by one
                self.further(depth=self.depth+1)
                if self.update_delta:
                    self.delta = self.delta_update()
                    if self.verbose > 0:
                        print 'delta was fitted to', self.delta
                if self.verbose > 0:
                    print 'depth is:', self.depth
                oob_err = self.get_oob_error(self.depth)
                if self.verbose > 0:
                    print 'oob error is:', oob_err 
                if oob_err < self.min_oob_err:
                    self.min_oob_err = oob_err
                    self.opt_depth = self.depth
                # decide whether tree needs to be furthered
                grow_further = (curr_depth < self.depth) and (self.depth < self.max_depth) 
                if self.build_to_opt_depth and (self.depth >= self.min_depth):
                    grow_further = grow_further and (oob_err == self.min_oob_err)
                    pass
                curr_depth = self.depth
        #####################################################
        #Growing full tree one by one
        else:
            self.further(depth=self.max_depth)
        return self
    
    def clear(self):
        if self.tc != None:
            parMixedForest.clearMemory(self.tc)
    
    def tree_sample(self):
        if self.subsampling:
            n_sample = SP.int_(self.n*self.sampsize)
            subsample = SP.random.permutation(self.n)[:n_sample]
        else:
            subsample = SP.random.random_integers(0,self.n-1, self.n)
        return subsample
            
    def further(self, depth=float('inf')):
        if self.tc == None:
            for i in SP.arange(len(self.trees)):
                if self.verbose > 1:
                    print 'growing tree ', i
                self.trees[i].grow(depth)
                if depth == float('inf'): #clear memory
                    self.trees[i].clear_data()
                    
        else:
            self.depth = parMixedForest.par_further(self.tc, depth)   
    
    def _predict(self, X=None, depth=float('inf'), oob=False, conf=False):
        if X == None:
            n_res = self.n
        else:
            n_res = X.shape[0]
        response = SP.zeros(n_res)
        count = SP.zeros(n_res)
        for tree in self.trees:
            if oob:
                response[tree.oob] += tree.predict(depth=depth, oob=oob, conf=conf)
                count[tree.oob] += 1
            elif X==None:
                response[tree.subsample] += tree.predict(depth=depth, oob=oob, conf=conf)
                count[tree.subsample] +=1
            else:
                response += tree.predict(X=X, depth=depth, conf=conf)
                count += 1
        response[count!=0] /= count[count!=0]
        if (count == 0).sum() > 1:
            warnings.warn('not all samples have been used for learning/ please use more trees')  
            response[count==0] = SP.mean(response)
        return response
    
    def get_oob_error(self, depth):
        return ((self._predict(depth=depth, conf=True, oob=True)-self.y.reshape(-1))**2).sum()/self.n
        
    def get_oob_likelihood(self, depth):
        oob_fixed_effect = self._predict(depth=depth, oob=True).reshape(-1,1)
        return self.BLUP.LL(oob_fixed_effect)
    
    def get_training_likelihood(self, depth):
        oob_fixed_effect = self._predict(depth=depth, oob=False).reshape(-1,1)
        return self.BLUP.LL(oob_fixed_effect)
    
    def delta_update(self):
        mean = self._predict(depth=self.depth, oob=False)
        self.BLUP.update_delta(mean)
        return self.BLUP.delta
        
    
    def predict(self, X, k=None, depth=None):
        """Predict response for X.

        The response to an input sample is computed as the sum of 
        (1) the mean prediction of the trees in the forest (fixed effect) and 
        (2) the estimated random effect. 

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.
        k: array-like of shape = [n_samples, n_samples_fitting]
            The cross-dependency structure between the samples used for learning the forest and the input samples.
            If not specified only the estimated fixed effect is returned.
            
        Returns
        -------
        y : array of shape = [n_samples, 1]
            The response
        """
        if depth == None:
            if self.opt_depth != None:
                depth = self.opt_depth
                if self.verbose > 0:
                    'using optimal depth to predict'
            else:
                depth = float('inf')
        response = self._predict(X=X, depth=depth)
        if k!=None:
                mean = self.predict(X=self.X, depth=depth).reshape(-1,1)
                response += self.BLUP.predict(XTest=X, k=k, mean=mean).reshape(-1)
        return response
    
if __name__== '__main__':
    mf = Forest()
    import pickle
    pickle.dump(mf,open('new_file', 'w'))
