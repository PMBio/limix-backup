def getClient():
    from IPython.parallel import Client
    import os
    sys_name = os.uname()[1]
    if sys_name[:3] == 'ebi':
        return Client(profile='lsf')
    else:
        return Client()


def setPaths(projects_path='~/my_research/region_testing'):
    import sys
    import os
    paths = []
    modulePath = os.path.expanduser(projects_path)
    paths.append(modulePath)
    sys.path.extend(paths)
    return sys.path
    
def par_init(tc, **rem_init_parameters):
    print(('starting mixed forest on', len(tc), 'nodes'))
    init_parameters = rem_init_parameters.copy()
    n_trees = [init_parameters['n_estimators'] // len(tc)] * len(tc)
    for i in range(init_parameters['n_estimators'] % len(tc)):
        n_trees[i] += 1
    dview = tc[:]
    dview.block = True
    dview.apply(setPaths)[0]
    dview.execute('from mixed_forest import MixedForest')
    dview.execute('from mixed_forest.parMixedForest import *')
    lbview = tc.load_balanced_view()
    jobs = []
    for n_tree in n_trees:
        init_parameters['n_estimators'] = n_tree
        job = lbview.apply_async(init, *[init_parameters])
        jobs.append(job)
    tc.wait(jobs)
    
def init(init_parameters):
    from mixed_forest import MixedForest
    globals()['forest']= MixedForest.Forest(**init_parameters)
    return True

def par_build_stumps(tc, X, y, kernel, delta):
    dview = tc[:]
    dview.block = True
    jobs = dview.apply(build_stumps, *[X, y, kernel, delta])
    #for job in jobs:
        #print job

def build_stumps(X, y, kernel, delta):
    globals()['forest'].build_stumps(X, y, kernel, delta)
    return True

def par_fixed_effect(tc, X, oob, depth):
    import scipy as SP
    dview = tc[:]
    dview.block = True
    results = dview.apply(fixed_effect, *[X, oob, depth])
    fixed_sum = SP.zeros_like(results[0][0])
    count = SP.zeros_like(results[0][1])
    for res in results:
        fixed_sum += res[0]
        count += res[1]
    return fixed_sum, count

def clearMemory(tc):
    dview = tc[:]
    dview.block=True
    dview.apply(clearall)
    dview.results.clear()
    tc.metadata.clear()
    tc.results.clear()

def clearall():
    """clear all globals"""
    for uniquevar in [var for var in globals().copy() if var[0] != "_" and var != 'clearall']:
        del globals()[uniquevar]
        
def fixed_effect(X, oob, depth):
    return globals()['forest'].__fixed_effect__(X, oob, depth)

def par_further(tc, depth):
    import scipy as SP
    dview = tc[:]
    dview.block = True
    depths = SP.array(dview.apply(further, *[depth]))
    return depths[SP.argmax(depths)]
     
def further(depth):
    globals()['forest'].further(depth)
    return globals()['forest'].depth

def par_update_delta(tc, delta):
    dview = tc[:]
    dview.block = True
    dview.apply(update_delta, *[delta])
    
def update_delta(delta):
    globals()['forest'].delta = delta
    return True

def par_get_variable_scores(tc):
    import scipy as SP
    dview = tc[:]
    dview.block = True
    results = dview.apply(get_variable_scores)
    var_used = SP.zeros_like((results[0])[0])
    log_importance = SP.zeros_like(var_used)
    for result in results:
        var_used +=  result[0]
        log_importance += result[1]
    return var_used, log_importance
        
def get_variable_scores():
    return globals()['forest'].var_used, globals()['forest'].log_importance