import inspect
import os

# HOW TO USE: Please, look at the end of the file for an example.

class Cached(object):
    def __init__(self):
        pass

    def clear_cache(self, *method_names):
        for method_name in method_names:
            setattr(self, '_cache_' + method_name, None)
            setattr(self, '_cached_' + method_name, False)
            setattr(self, '_cached_args_' + method_name, dict())

# This decorator works both with arguments or without
# (i.e., @cached or @cached(exclude=['param1', 'param2', ...]))
def cached(*args, **kwargs):

    # - len(args) == 0 means decorator with arguments, which are stored in
    #   kwargs
    # - (len(args) == 1 and inspect.isfunction(args[0])) means decorator
    #   without arguments, and args[0] provides the method to be decorated
    assert len(args) == 0 or (len(args) == 1 and inspect.isfunction(args[0]))
    deco_without_arg = len(args) == 1

    # cached_args stores the argument names and argument values with which
    # the method was called in
    # cached_args = dict()
    if deco_without_arg:
        filter_out = []
    else:
        assert len(kwargs) == 1 and 'exclude' in kwargs
        filter_out = kwargs['exclude']

    def real_cached(method):
        cache_var_name = '_cache_' + method.__name__
        valid_var_name = '_cached_' + method.__name__
        cached_args_name = '_cached_args_' + method.__name__

        debug = int(os.getenv('LIMIX_DEBUG', 0))
        if debug:
            return method

        def method_wrapper(self, *args, **kwargs):
            (argnames, argvalues) = _fetch_argnames_argvalues(method, args, kwargs)

            t = zip(argnames, argvalues)
            provided_args = dict((x, y) for x, y in t)

            for f in filter_out:
                del provided_args[f]

            if getattr(self, valid_var_name, False) is False or\
                    provided_args != getattr(self, cached_args_name, dict()):

                result = method(self, *args, **kwargs)
                setattr(self, cache_var_name, result)
                setattr(self, valid_var_name, True)
                if hasattr(self, cached_args_name):
                    getattr(self, cached_args_name).clear()
                    getattr(self, cached_args_name).update(provided_args)
                else:
                    setattr(self, cached_args_name, dict())

            return getattr(self, cache_var_name)

        return method_wrapper

    if deco_without_arg:
        return real_cached(args[0])

    return real_cached


########################### INTERNAL USE ONLY ###########################


def _map_args_kwargs_to_argvalues(args, kwargs, argnames, defaults):

    argvalues = [None] * len(argnames)
    for i in xrange(len(defaults)):
        argvalues[len(argnames) - len(defaults) + i] = defaults[i]

    for i in xrange(len(args)):
        argvalues[i] = args[i]

    for kw in kwargs:
        index = argnames.index(kw)
        argvalues[index] = kwargs[kw]

    return argvalues

# This function retrieves and organizes the method argument names and
# their values at the moment of the call. For this task,
# I need to both inspect the method (looking for default values) and
# the actual params passed by the user. Combining these two source of
# information I can return (argnames, argvalues) with all the argument names
# and argument values.
def _fetch_argnames_argvalues(method, args, kwargs):
    argnames = inspect.getargspec(method)[0]
    del argnames[argnames.index('self')]

    defaults = inspect.getargspec(method)[3]
    if defaults is None:
        defaults = []
    argvalues = _map_args_kwargs_to_argvalues(args, kwargs, argnames, defaults)

    return (argnames, argvalues)

if __name__ == '__main__':
    class Test(Cached):
        @cached
        def foo1(self, par1, par2=None):
            return 5

        @cached(exclude=['par2'])
        def foo2(self, par1, par2):
            return 5

    test = Test()
    print 'test.foo1(2, par2=0)',
    test.foo1(2, par2=0)
    print 'test.foo1(2, par2=0)',
    test.foo1(2, par2=0)
    print 'test.foo1(2, par2=0)',
    test.foo1(2, par2=0)
    print 'test.foo1(2, par2=3)',
    test.foo1(2, par2=3)
    print 'test.foo1(2, par2=3)',
    test.foo1(2, par2=3)
    print 'test.foo1(2)',
    test.foo1(2)
    print 'test.foo1(3)',
    test.foo1(3)
    print 'test.foo1(3)',
    test.foo1(3)

    print 'test.foo2(1, 2)',
    test.foo2(1, 2)
    print 'test.foo2(1, 2)',
    test.foo2(1, 2)
    print 'test.foo2(1, 3)',
    test.foo2(1, 3)
    print 'test.foo2(2, 3)',
    test.foo2(2, 3)
    print 'test.foo2(2, 3)',
    test.foo2(2, 3)
