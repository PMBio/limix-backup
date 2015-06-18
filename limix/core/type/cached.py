import inspect
import os

# HOW TO USE: Please, look at the end of the file for an example.

# This is a hacky thing so that one can know that a method_wrapper object,
# defined bellow, really came from here.
_cache_identifier = 'pj97YCjgnp'


class Cached(object):

    def __init__(self):
        self._cache_groups = dict()

        for (_, m) in inspect.getmembers(type(self),
                                         predicate=inspect.ismethod):
            pass
            try:
                fv = m.im_func.func_code.co_freevars
            except AttributeError:
                continue
            closure = m.im_func.func_closure
            if closure is None:
                continue

            # vs = dict(zip(fv, (c.cell_contents for c in closure)))
            # groups = vs['groups']
            # import ipdb; ipdb.set_trace()
            # for g in groups:
            #     self._cache_groups[g] = []

    def clear_cache(self, *cache_groups):
        assert hasattr(self, '_cache_groups'), "Cache system failed because "\
                                   + "you did not call Cached's constructor."

        for g in cache_groups:
            if g not in self._cache_groups:
                continue
            #     err = 'Cache group %s does not exist.' % g
            #     err += ' It might happen because of two reasons. Either'
            #     err += ' you never defined this group or'
            #     err += ' you are trying to clear a cache group that'
            #     err += ' has not been used yet.'
            #     raise Exception(err)

            for method_name in self._cache_groups[g]:
                setattr(self, '_cache_' + method_name, None)
                setattr(self, '_cached_' + method_name, False)
                setattr(self, '_cached_args_' + method_name, dict())

    def fill_cache(self, method_name, value):
        setattr(self, '_cache_' + method_name, value)
        setattr(self, '_cached_' + method_name, True)


# This decorator works both with arguments or without
# (i.e., @cached, @cached("group_name"), @cached(["group_name_A",
#       "group_name_B"]), or @cached(exclude=['param1', 'param2', ...]))
def cached(*args, **kwargs):

    deco_without_arg = len(args) == 1 and inspect.isfunction(args[0])

    # cached_args stores the argument names and argument values with which
    # the method was called in
    # cached_args = dict()
    filter_out = []
    groups = ['default']

    cache_identifier = _cache_identifier

    if not deco_without_arg:
        if len(args) > 0:
            if type(args[0]) is list or type(args[0]) is tuple:
                groups += list(args[0])
            else:
                groups.append(args[0])

        if len(kwargs) == 1:
            assert 'exclude' in kwargs, ("'exclude' is the only keyword "
                                         "allowed here.")
            filter_out = kwargs['exclude']
    else:
        groups.append(args[0].__name__)

    def real_cached(method):
        cache_var_name = '_cache_' + method.__name__
        valid_var_name = '_cached_' + method.__name__
        cached_args_name = '_cached_args_' + method.__name__

        debug = int(os.getenv('LIMIX_DEBUG', 0))
        if debug:
            return method

        def method_wrapper(self, *args, **kwargs):

            (argnames, argvalues) = _fetch_argnames_argvalues(method, args,
                                                              kwargs)

            t = zip(argnames, argvalues)
            provided_args = dict((x, y) for x, y in t)

            for f in filter_out:
                del provided_args[f]

            for g in groups:
                if g not in self._cache_groups:
                    self._cache_groups[g] = []

                if method.__name__ not in self._cache_groups[g]:
                    self._cache_groups[g].append(method.__name__)

            if not hasattr(self, cached_args_name):
                setattr(self, cached_args_name, dict())

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


# ------------------------ INTERNAL USE ONLY ------------------------


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
            print 'called',
            return 5

        @cached(exclude=['par2'])
        def foo2(self, par1, par2):
            print 'called',
            return 5

        @cached("A")
        def foo3(self, par1):
            print 'called',
            return 5

        @cached(["A", "B"])
        def foo4(self, par1):
            print 'called',
            return 5

        @cached(["B"])
        def foo5(self, par1):
            print 'called',
            return 5

    test = Test()
    print '\ntest.foo1(2, par2=0)',
    test.foo1(2, par2=0)
    print '\ntest.foo1(2, par2=1)',
    test.foo1(2, par2=1)
    print '\ntest.foo1(2, par2=0)',
    test.foo1(2, par2=0)
    print '\ntest.foo1(2, par2=3)',
    test.foo1(2, par2=3)
    print '\ntest.foo1(2, par2=3)',
    test.foo1(2, par2=3)
    print '\ntest.foo1(2)',
    test.foo1(2)
    print '\ntest.foo1(3)',
    test.foo1(3)
    print '\ntest.foo1(3)',
    test.foo1(3)

    print ''

    print '\ntest.foo2(1, 2)',
    test.foo2(1, 2)
    print '\ntest.foo2(1, 2)',
    test.foo2(1, 2)
    print '\ntest.foo2(1, 3)',
    test.foo2(1, 3)
    print '\ntest.foo2(2, 3)',
    test.foo2(2, 3)
    print '\ntest.foo2(2, 3)',
    test.foo2(2, 3)

    print ''
    print '\ntest.foo3(1)',
    test.foo3(1)
    print '\ntest.foo3(1)',
    test.foo3(1)
    print '\ntest.foo4(1)',
    test.foo4(1)
    print '\ntest.foo4(1)',
    test.foo4(1)
    print '\ntest.foo5(1)',
    test.foo5(1)
    print '\ntest.foo5(1)',
    test.foo5(1)
    print '\nClearing cache group A',
    test.clear_cache("A")
    print '\ntest.foo3(1)',

    test.foo3(1)
    print '\ntest.foo3(1)',
    test.foo3(1)
    print '\ntest.foo4(1)',
    test.foo4(1)
    print '\ntest.foo4(1)',
    test.foo4(1)
    print '\ntest.foo5(1)',
    test.foo5(1)
    print '\ntest.foo5(1)',
    test.foo5(1)
    print '\nClearing cache group B',
    test.clear_cache("B")
    print '\ntest.foo3(1)',
    test.foo3(1)
    print '\ntest.foo3(1)',
    test.foo3(1)
    print '\ntest.foo4(1)',
    test.foo4(1)
    print '\ntest.foo4(1)',
    test.foo4(1)
    print '\ntest.foo5(1)',
    test.foo5(1)
    print '\ntest.foo5(1)',
    test.foo5(1)
    print '\nClearing cache group default',
    test.clear_cache("default")
    print '\ntest.foo3(1)',
    test.foo3(1)
    print '\ntest.foo3(1)',
    test.foo3(1)
    print '\ntest.foo4(1)',
    test.foo4(1)
    print '\ntest.foo4(1)',
    test.foo4(1)
    print '\ntest.foo5(1)',
    test.foo5(1)
    print '\ntest.foo5(1)',
    test.foo5(1)
