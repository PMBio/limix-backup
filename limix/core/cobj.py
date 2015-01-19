class cObject(object):
    """
    This class add to object a util function for caching
    """
    def __init__(self):
        pass

    def clear_cache(self,*method_names):
        for method_name in method_names:
            setattr(self,'_cache_'+method_name,None)
            setattr(self,'_cached_'+method_name,False)

def cached(method):
    """ this function is used as a decorator for caching """
    _cache_attr_name = '_cache_'+method.__name__
    _bool_attr_name  = '_cached_'+method.__name__
    def method_wrapper(self,*args,**kwargs):
        is_cached = getattr(self,_bool_attr_name)
        if not is_cached:
            result = method(self, *args, **kwargs)
            setattr(self, _cache_attr_name, result)
            setattr(self, _bool_attr_name, True)
        return getattr(self,'_cache_'+method.__name__)
    return method_wrapper

