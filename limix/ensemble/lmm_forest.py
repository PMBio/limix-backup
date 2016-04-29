'''
Created on Apr 2, 2013

@author: johannes
'''

import scipy as SP
import scipy.linalg as LA
import warnings
from . import py_splitting_core as SC
from . import par_lmm_forest
from . import lmm_forest_utils as utils
from . import blup as BLUP
from limix.ensemble import SplittingCore as CSP

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
                 optimize_memory_use=True,
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
        that fits a number of linear mixed decision trees on various sub-samples
        of the data set and uses averaging
        to improve the predictive accuracy and control over-fitting.

        Parameters
        ----------

        kernel: accepts either
            (1) positive definite matrix of size n x n, describing a dependency
                structure between the n samples (e.g. population structure).
            (2) strings 'data' or 'iid'
            If set to 'data' dependency structure is estimated from the
            predictor matrix X, if set to 'iid' i.i.d. samples are assumed
            and the linear mixed forest behaves like the random forest.

        n_estimators : integer, optional (default=100)
            The number of trees in the forest.

        ratio_features : fraction of the total number of features used for each
                         split (default=0.66)

        max_depth : optional (default='inf') real value that defining the
                    maximum depth of the tree.

        min_samples_split : integer, optional (default=5)
            The minimum number of samples required to split a node.

        fit_opt_depth: uses the forest likelihood to fit the optimal depth of
                       the trees in the ensemble on the out of bag samples
                       (default=False)

        build_to_opt_depth: stops furthering trees in the ensemble if
                            improvement of forest likelihood stops
                            (default=False). This helps to improve runtime
                            and to avoid overfitting when a relatively small
                            number of trees is used on a big dataset.

        sampsize: The number of samples drawn without replacement for growing a
                  single tree (default=0.5)

        subsampling: If 'True' this toggles sampling without replacement.
                     Otherwise as sampling without replacement is used
                     (default=True)

        verbose : int, optional (default=0) Controls the verbosity level of the
                  tree building process.
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

        # Settings for (re)initialization#
        self.X = None
        self.y = None
        self.n = None
        self.m = None
        self.BLUP = None
        self.max_features = None
        self.var_used = None
        self.log_importance = None
        self.trees = []

        if random_state is not None:
            SP.random.seed(random_state)
        if tc is not None:
            parMixedForest.par_init(**self.init_parameters)

    def reset(self):
        # Reinstatiate using the same parameters
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
        if isinstance(self.kernel, str) and self.kernel == 'data':
            self.kernel = SC.estimateKernel(X, maf=1.0/X.shape[0])
        elif isinstance(self.kernel, str) and self.kernel == 'iid':
            self.kernel = SP.identity(X.shape[0])
        # Use dedicated part of data as background model
        elif self.kernel.size == X.shape[1]:
            tmp_ind = self.kernel
            self.kernel = utils.estimateKernel(X[:, self.kernel],
                                               maf=1.0/X.shape[0])
            X = X[:, ~tmp_ind]
        # Extract and reshape data
        self.y = y.reshape(-1, 1)
        self.X = X
        self.n, self.m = self.X.shape
        if self.delta is None:
            self.BLUP = BLUP.BLUP()
            if self.verbose > 1:
                print('fitting BLUP')
            self.BLUP.fit(XTrain=self.X, yTrain=self.y, KTrain=self.kernel,
                          delta=self.delta)
            if self.verbose > 1:
                print('done fitting BLUP')
            # Update delta if it used to be 'None'
            self.delta = self.BLUP.delta
        self.max_features = SP.maximum(SP.int_(self.ratio_features*self.m), 1)
        self.var_used = SP.zeros(self.m)
        self.log_importance = SP.zeros(self.m)
        self.depth = 0

        if self.verbose > 0:
            print(('log(delta) fitted to ', SP.log(self.delta)))

        # Initialize individual trees
        if recycle and self.trees != []:
            for tree in self.trees:
                tree.cut_to_stump()
        else:
            n_trees = 0
            self.trees = []
            while n_trees < self.n_estimators:
                if self.verbose > 1:
                    print(('init. tree number ', n_trees))
                subsample = self.tree_sample()
                tree = MixedForestTree(self, subsample)
                self.trees.append(tree)
                n_trees += 1

        # Fitting with optimal depth constraint
        if self.fit_optimal_depth or self.update_delta:
            self.opt_depth = 0
            self.min_oob_err = self.get_oob_error(self.depth)
            if self.verbose > 0:
                print(('initial oob error is:', self.min_oob_err))
            grow_further = True
            curr_depth = self.depth
            while grow_further:
                # Updating ensemble increasing its depth by one
                self.further(depth=self.depth+1)
                if self.update_delta:
                    self.delta = self.delta_update()
                    if self.verbose > 0:
                        print(('delta was fitted to', self.delta))
                if self.verbose > 0:
                    print(('depth is:', self.depth))
                oob_err = self.get_oob_error(self.depth)
                if self.verbose > 0:
                    print(('oob error is:', oob_err))
                if oob_err < self.min_oob_err:
                    self.min_oob_err = oob_err
                    self.opt_depth = self.depth
                # Decide whether tree needs to be furthered
                grow_further = (curr_depth < self.depth) and\
                    (self.depth < self.max_depth)
                if self.build_to_opt_depth and (self.depth >= self.min_depth):
                    grow_further = grow_further and\
                        (oob_err == self.min_oob_err)
                    pass
                curr_depth = self.depth
        #####################################################
        # Growing full tree one by one
        else:
            self.further(depth=self.max_depth)
        return self

    def clear(self):
        if self.tc is not None:
            parMixedForest.clearMemory(self.tc)

    def tree_sample(self):
        if self.subsampling:
            n_sample = SP.int_(self.n*self.sampsize)
            subsample = SP.random.permutation(self.n)[:n_sample]
        else:
            subsample = SP.random.random_integers(0, self.n-1, self.n)
        return subsample

    def further(self, depth=float('inf')):
        if self.tc is None:
            for i in SP.arange(len(self.trees)):
                if self.verbose > 1:
                    print(('growing tree ', i))
                self.trees[i].grow(depth)
                if depth == float('inf'):
                    self.trees[i].clear_data()

        else:
            self.depth = parMixedForest.par_further(self.tc, depth)

    def _predict(self, X=None, depth=float('inf'), oob=False, conf=False):
        if X is None:
            n_res = self.n
        else:
            n_res = X.shape[0]
        response = SP.zeros(n_res)
        count = SP.zeros(n_res)
        for tree in self.trees:
            if oob:
                response[tree.oob] += tree.predict(depth=depth, oob=oob,
                                                   conf=conf)
                count[tree.oob] += 1
            elif X is None:
                response[tree.subsample] += tree.predict(depth=depth, oob=oob,
                                                         conf=conf)
                count[tree.subsample] += 1
            else:
                response += tree.predict(X=X, depth=depth, conf=conf)
                count += 1
        response[count != 0] /= count[count != 0]
        if (count == 0).sum() > 1:
            warnings.warn('not all samples have been used for learning\
                          please use more trees')
            response[count == 0] = SP.mean(response)
        return response

    def get_oob_error(self, depth):
        return ((self._predict(depth=depth, conf=True, oob=True) -
                 self.y.reshape(-1))**2).sum()/self.n

    def get_oob_likelihood(self, depth):
        oob_fixed_effect = self._predict(depth=depth, oob=True).reshape(-1, 1)
        return self.BLUP.LL(oob_fixed_effect)

    def get_training_likelihood(self, depth):
        oob_fixed_effect = self._predict(depth=depth, oob=False).reshape(-1, 1)
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
            The cross-dependency structure between the samples used for learning
            the forest and the input samples.
            If not specified only the estimated fixed effect is returned.

        Returns
        -------
        y : array of shape = [n_samples, 1]
            The response
        """
        if depth is None:
            if self.opt_depth is not None:
                depth = self.opt_depth
                if self.verbose > 0:
                    'using optimal depth to predict'
            else:
                depth = float('inf')
        response = self._predict(X=X, depth=depth)
        if k is not None:
                mean = self.predict(X=self.X, depth=depth).reshape(-1, 1)
                response += self.BLUP.predict(XTest=X, k=k,
                                              mean=mean).reshape(-1)
        return response


class MixedForestTree(object):
    '''
    classdocs
    '''
    def __init__(self, forest, subsample):
        '''
        Constructor
        '''
        self.forest = forest
        if self.forest is not None:
            self.verbose = self.forest.verbose
        else:
            self.verbose = 0
        self.max_depth = 0
        # Estimate the potential number of nodes
        self.subsample = subsample
        self.subsample_bin = SP.zeros(self.forest.n, dtype='bool')
        self.subsample_bin[self.subsample] = True
        self.oob = SP.arange(self.forest.n)[~self.subsample_bin]
        nr_nodes = 4*subsample.size
        self.nodes = SP.zeros(nr_nodes, dtype='int')
        self.best_predictor = SP.empty(nr_nodes, dtype='int')
        self.start_index = SP.empty(nr_nodes, dtype='int')
        self.end_index = SP.empty(nr_nodes, dtype='int')
        self.left_child = SP.zeros(nr_nodes, dtype='int')
        self.right_child = SP.zeros(nr_nodes, dtype='int')
        self.parent = SP.empty(nr_nodes, dtype='int')
        self.mean = SP.zeros(nr_nodes)
        # Initialize root node
        self.node_ind = 0
        self.nodes[self.node_ind] = 0
        self.start_index[self.node_ind] = 0
        self.end_index[self.node_ind] = subsample.size
        self.num_nodes = 1
        self.num_leafs = 0
        self.s = SP.ones_like(self.nodes)*float('inf')
        kernel = self.get_kernel()
        if not(self.forest.optimize_memory_use):
            self.X = self.forest.X[subsample]
        if self.verbose > 1:
            print('compute tree wise singular value decomposition')
        self.S, self.U = LA.eigh(kernel + SP.eye(subsample.size)*1e-8)
        self.Uy = SP.dot(self.U.T, self.forest.y[subsample])
        if self.verbose > 1:
            print('compute tree wise bias')
        self.mean[0] = SC.estimate_bias(self.Uy, self.U, self.S,
                                        SP.log(self.forest.delta))
        self.sample = SP.arange(subsample.size)
        ck = self.get_cross_kernel(self.oob, self.subsample)
        self.cross_core = SP.dot(ck, LA.inv(kernel +
                                            SP.eye(self.subsample.size) *
                                            self.forest.delta))
        if self.verbose > 1:
            print('done initializing tree')


    def print_tree(self):
        return self.print_tree_rec(0)

    def print_tree_rec(self, node_ind):
        if SC.is_leaf(node_ind, self.left_child):
            return SP.str_(self.mean[node_ind])
        else:
            left_child = self.left_child[node_ind]
            right_child = self.right_child[node_ind]
            return '(' + self.print_tree_rec(left_child) + ','\
                + self.print_tree_rec(right_child) + ')'

    def grow(self, depth=float('inf')):
        while (self.node_ind < self.num_nodes) and\
              (self.get_depth(self.nodes[self.node_ind]) < depth) and\
              (self.num_leafs <= self.forest.max_nodes-2):
            if (self.end_index[self.node_ind] -
               self.start_index[self.node_ind]) > self.forest.min_samples_split:

                best_j, best_s, left_mean, right_mean, ll_score =\
                    self.split_node(self.node_ind)

                if best_j != -1:
                    self.best_predictor[self.node_ind] = best_j
                    self.s[self.node_ind] = best_s
                    self.forest.var_used[best_j] += 1
                    self.forest.log_importance[best_j] += ll_score
                    split = self.rearrange(best_j, best_s, self.node_ind)
                    self.nodes[self.num_nodes] = self.nodes[self.node_ind]*2 + 1
                    self.nodes[self.num_nodes+1] =\
                        self.nodes[self.node_ind]*2 + 2
                    self.mean[self.num_nodes] = left_mean
                    self.mean[self.num_nodes+1] = right_mean
                    self.start_index[self.num_nodes] =\
                        self.start_index[self.node_ind]
                    self.start_index[self.num_nodes+1] = split
                    self.end_index[self.num_nodes] = split
                    self.end_index[self.num_nodes+1] =\
                        self.end_index[self.node_ind]
                    self.left_child[self.node_ind] = self.num_nodes
                    self.right_child[self.node_ind] = self.num_nodes + 1
                    self.parent[self.num_nodes] = self.node_ind
                    self.parent[self.num_nodes+1] = self.node_ind
                    self.forest.depth =\
                        SP.maximum(self.get_depth(self.nodes[self.num_nodes]),
                                   self.forest.depth)
                    if self.forest.depth > 50:
                        warnings.warn('tree depth exceeded 50, check for\
                                      imbalanced predictors and consider to\
                                      restrict the maximum depth of trees')
                    self.num_nodes += 2
                    self.num_leafs += 1
            self.node_ind += 1

    def get_depth(self, node):
            depth = SP.log2(node+1)
            return SP.floor(depth)

    def get_kernel(self):
        return self.forest.kernel[SP.ix_(self.subsample, self.subsample)]

    def get_cross_kernel(self, oob, subsample):
        return (self.forest.kernel +
                self.forest.delta*SP.eye(self.forest.n))[SP.ix_(oob, subsample)]

    def predict(self, X=None, depth=float('inf'), oob=False, conf=False):
        ##################################################
        # getting correct X
        if X is None:
            if oob:
                X = self.forest.X[self.oob]
            else:
                X = self.get_X()
        ##################################################
        # making prediction
        if self.forest.cpp_predict:
            response = self.predict_cpp(X, depth)
        else:
            response = self.predict_py(X, depth)
        if conf:
            mean = self.predict(self.forest.X[self.subsample], depth=depth)
            # TODO this dot product can be cached
            response += SP.dot(self.cross_core,
                               self.forest.y[self.subsample].reshape(-1)-mean)
        return response

    def predict_cpp(self, X, depth):
        #ensure consistency to c data type
        X = SP.array(X, dtype='float')
        response = SP.empty(X.shape[0])
        return CSP.predict(response, self.nodes, self.left_child,
                           self.right_child, self.best_predictor, self.mean,
                           self.s, X, depth)

    def predict_py(self, X, depth):
        response = SP.empty(X.shape[0])
        for i in SP.arange(X.shape[0]):
            response[i] = self.predict_rec(0, X[i, :], depth)
        return response

    def predict_rec(self, node_ind, x, depth):
        if self.get_depth(self.nodes[node_ind]) == depth or SC.is_leaf(node_ind,
                          self.left_child):
            return self.mean[node_ind]
        else:
            if x[self.best_predictor[node_ind]] < self.s[node_ind]:
                # Go to left child node
                return self.predict_rec(self.left_child[node_ind], x, depth)
            else:
                # Else go to right child node
                return self.predict_rec(self.right_child[node_ind], x, depth)

    def rearrange(self, j, best_s, node_ind):
        node_start = self.start_index[node_ind]
        node_end = self.end_index[node_ind]
        nodesamples = self.sample[node_start:node_end]
        left_indexes = self.get_X_slice(j)[nodesamples] < best_s
        # Print left_indexes
        left_samples = nodesamples[left_indexes]
        right_samples = nodesamples[~left_indexes]
        left_end = node_start+left_indexes.sum()
        # Resort to make split possible
        self.sample[node_start:node_end] =\
            SP.hstack((left_samples, right_samples))
        return left_end

    def split_node(self, node_ind):
        noderange = self.sample[self.start_index[node_ind]:self.end_index[node_ind]]
        rmind = SP.random.permutation(self.forest.m)[:self.forest.max_features]
        mBest = -1
        sBest = -float('inf')
        left_mean = None
        right_mean = None
        ll_score = -float('inf')
        if rmind.size > 10 and self.forest.optimize_memory_use:
            n_slice = 10
        else:
            n_slice = 1
        intv = SP.floor(n_slice*SP.arange(rmind.size)/rmind.size)
        for slc in SP.arange(n_slice):
            rmind_slice = rmind[intv==slc]
            X = self.get_X_slice(rmind_slice)
            Covariates = SC.get_covariates(node_ind,
                                self.nodes[node_ind],
                                self.parent,
                                self.sample,
                                self.start_index,
                                self.end_index)

            if self.forest.cpp_fit:
                    mBest_, sBest_, left_mean_, right_mean_, ll_score_ =\
                        SC.cpp_best_split_full_model(X, self.Uy, Covariates,
                                                     self.S,
                                                     self.U, noderange,
                                                     self.forest.delta)
            else:
                mBest_, sBest_, left_mean_, right_mean_, ll_score_ =\
                    SC.best_split_full_model(X, self.Uy, Covariates, self.S,
                                             self.U, noderange,
                                             self.forest.delta)

            if (mBest_ != -1) and (ll_score_ > ll_score):
                mBest = rmind_slice[mBest_]
                sBest = sBest_
                left_mean = left_mean_
                right_mean = right_mean_
                ll_score = ll_score_
        return mBest, sBest, left_mean, right_mean, ll_score

    def cut_to_stump(self):
        self.max_depth = 0
        self.node_ind = 0
        self.nodes[self.node_ind] = 0
        self.start_index[self.node_ind] = 0
        self.end_index[self.node_ind] = self.subsample.size
        self.num_nodes = 1
        self.num_leafs = 0
        self.left_child = SP.zeros_like(self.left_child)
        self.right_child = SP.zeros_like(self.right_child)

    def clear_data(self):
        '''
        Free memory
        If many trees are grown this is an useful options since it is saving a
        lot of memory '''
        if self.forest.verbose > 1:
            print('clearing up stuff')
        self.S = None
        self.Uy = None
        self.U = None

    def get_X(self):
        if self.forest.optimize_memory_use:
            return self.forest.X[self.subsample]
        else:
            return self.X

    def get_X_slice(self, rmind):
        '''get X with slicing'''
        if self.forest.optimize_memory_use:
            return self.forest.X[:, rmind][self.subsample]
        else:
            return self.X[:, rmind]
