'''
Created on Apr 2, 2013

@author: johannes
'''
import scipy as SP
import scipy.linalg as LA
import pySplittingCore as SC
import SplittingCore as CSP
import warnings
class MixedForestTree(object):
    '''
    classdocs
    '''
    def __init__(self, forest, subsample):
        '''
        Constructor
        '''
        self.forest = forest
        self.max_depth = 0
        # Estimate the potential number of nodes
        self.subsample = subsample
        self.subsample_bin = SP.zeros(self.forest.n, dtype='bool')
        self.subsample_bin[self.subsample] = True
        self.oob = SP.arange(self.forest.n)[~self.subsample_bin]
        #self.oob = self.subsample
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
            self.X = self.forest.X[subsample] #TODO try to save X more elegant
        self.S, self.U = LA.eigh(kernel + SP.eye(subsample.size)*1e-8)
        self.Uy = SP.dot(self.U.T, self.forest.y[subsample])
        self.mean[0] = SC.estimate_bias(self.Uy, self.U, self.S, SP.log(self.forest.delta))
        self.sample = SP.arange(subsample.size)
        ck = self.get_cross_kernel(self.oob, self.subsample)
        self.cross_core = SP.dot(ck, LA.inv(kernel + SP.eye(self.subsample.size)*self.forest.delta))
        
    def print_tree(self):
        return self.print_tree_rec(0)
        
    def print_tree_rec(self, node_ind):
        if SC.is_leaf(node_ind, self.left_child):
            return SP.str_(self.mean[node_ind])
        else:
            left_child = self.left_child[node_ind]
            right_child = self.right_child[node_ind]
            return '(' + self.print_tree_rec(left_child) + ',' +  self.print_tree_rec(right_child) + ')'
                
    def grow(self, depth=float('inf')):
        while (self.node_ind < self.num_nodes) and (self.get_depth(self.nodes[self.node_ind]) < depth) and (self.num_leafs <= self.forest.max_nodes-2):
            if (self.end_index[self.node_ind] - self.start_index[self.node_ind]) > self.forest.min_samples_split: 
                best_j, best_s, left_mean, right_mean, ll_score = self.split_node(self.node_ind)
                if best_j != -1:
                    self.best_predictor[self.node_ind] = best_j
                    self.s[self.node_ind] = best_s
                    self.forest.var_used[best_j] += 1
                    self.forest.log_importance[best_j] += ll_score
                    split = self.rearrange(best_j, best_s, self.node_ind)
                    self.nodes[self.num_nodes] = self.nodes[self.node_ind]*2 + 1
                    self.nodes[self.num_nodes+1] = self.nodes[self.node_ind]*2 + 2
                    self.mean[self.num_nodes] = left_mean
                    self.mean[self.num_nodes+1] = right_mean
                    self.start_index[self.num_nodes] = self.start_index[self.node_ind]
                    self.start_index[self.num_nodes+1] = split
                    self.end_index[self.num_nodes] = split
                    self.end_index[self.num_nodes+1] = self.end_index[self.node_ind]
                    self.left_child[self.node_ind] = self.num_nodes
                    self.right_child[self.node_ind] = self.num_nodes + 1
                    self.parent[self.num_nodes] = self.node_ind
                    self.parent[self.num_nodes+1] = self.node_ind
                    self.forest.depth = SP.maximum(self.get_depth(self.nodes[self.num_nodes]), self.forest.depth)
                    if self.forest.depth > 50:
                        warnings.warn('tree depth exceeded 50, check for imbalanced predictors and consider to restrict the maximum depth of trees')
                    self.num_nodes += 2
                    self.num_leafs += 1
            self.node_ind+=1
            
    def get_depth(self, node):
            depth = SP.log2(node+1)
            return SP.floor(depth)
        
    def get_kernel(self):
        return self.forest.kernel[SP.ix_(self.subsample,self.subsample)]
    
    def get_cross_kernel(self, oob, subsample):
        return (self.forest.kernel+self.forest.delta*SP.eye(self.forest.n))[SP.ix_(oob, subsample)]
    
    def predict(self, X=None, depth=float('inf'), oob=False, conf=False):
        ##################################################
        # getting correct X
        if X==None:
            if oob:
                X = self.forest.X[self.oob]
            else:
                X = self.get_self_X() 
        ##################################################
        # making prediction
        if self.forest.cpp_predict:
            response = self.predict_cpp(X, depth)
        else:
            response = self.predict_py(X,depth)
        if conf:
            mean = self.predict(self.forest.X[self.subsample], depth=depth)
            #TODO this dot product can be cached
            response += SP.dot(self.cross_core, self.forest.y[self.subsample].reshape(-1)-mean)
        return response
            
    def predict_cpp(self, X, depth):
        #ensure consistency to c data type
        X = SP.array(X, dtype='float')
        response = SP.empty(X.shape[0])
        return CSP.predict(response, self.nodes, self.left_child, self.right_child, self.best_predictor, self.mean, self.s, X, depth)
    
    def predict_py(self, X, depth):
        response = SP.empty(X.shape[0])    
        for i in SP.arange(X.shape[0]):
            response[i] = self.predict_rec(0, X[i,:], depth)
        return response
        
            
    def predict_rec(self, node_ind, x, depth):
        if self.get_depth(self.nodes[node_ind]) == depth or SC.is_leaf(node_ind, self.left_child):
            return self.mean[node_ind]
        else:
            if x[self.best_predictor[node_ind]] < self.s[node_ind]:
                return self.predict_rec(self.left_child[node_ind], x, depth) #go to left child node
            else:
                return self.predict_rec(self.right_child[node_ind], x, depth) #go to right child node
                
    def rearrange(self, j, best_s, node_ind):
        node_start = self.start_index[node_ind]
        node_end = self.end_index[node_ind]
        nodesamples = self.sample[node_start:node_end]
        left_indexes = self.get_self_X()[nodesamples, j] < best_s
        #print left_indexes
        left_samples = nodesamples[left_indexes]
        right_samples = nodesamples[~left_indexes]
        left_end = node_start+left_indexes.sum()
        # resort to make split possible
        self.sample[node_start:node_end] = SP.hstack((left_samples, right_samples))
        return left_end 
    
    def split_node(self, node_ind):
        noderange = self.sample[self.start_index[node_ind]:self.end_index[node_ind]]
        rmind = SP.random.permutation(self.forest.m)[:self.forest.max_features]  
        mBest = -1
        sBest = -float('inf')
        left_mean = None
        right_mean = None
        ll_score = None
        X = self.get_self_X()
        X = SP.array(X[:,rmind], order='C')
        if X.shape[1] > 0:
            Covariates = SC.get_covariates(node_ind, 
                                   self.nodes[node_ind], 
                                   self.parent, 
                                   self.sample, 
                                   self.start_index, 
                                   self.end_index)
            if self.forest.cpp_fit:
                mBest, sBest, left_mean, right_mean, ll_score = SC.cpp_best_split_full_model(X, self.Uy, Covariates, self.S, self.U, noderange, self.forest.delta)
            else:    
                mBest, sBest, left_mean, right_mean, ll_score = SC.best_split_full_model(X, self.Uy, Covariates, self.S, self.U, noderange, self.forest.delta)
            if mBest != -1:
                mBest = rmind[mBest]
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
        save memory
        if many trees are grown this is an useful options since it is saving a lot of memory '''
        if self.forest.verbose > 1:
            print 'clearing up stuff'
        #self.X = None
        self.S = None
        self.Uy = None
        self.U = None
        
    def get_self_X(self):
        if self.forest.optimize_memory_use:
            return self.forest.X[self.subsample]
        else: 
            return self.X
        
if __name__== '__main__':
    
    import MixedForest
    n = 20
    m = 5
    forest = MixedForest.Forest(n_estimators=10, random_state=5, min_samples_split=5)
    X = (SP.random.rand(n,m) < 0.5) * 1
    X[:,2] = 0
    y = SP.random.randn(n,1) + SP.random.randn(n,1)*X[:,0:1]
    forest.fit(X=X, y=y)
    x_predict = X[2:3,:]
    print forest.trees[0].predict(X[2:3,:], 10)
    print forest.predict(x_predict)
    print forest.trees[0].print_tree()
    print 'done'
