import inspect
import os
import pprint
import re
from time import time
from limix.utils.util_functions import smartSum


# HOW TO USE: Please, look at the end of the file for an example.

def std_obj_repr(obj):
    return "<%s instance at %s>" % (obj.__class__.__name__, hex(id(obj)))

class Cached(object):

    def __init__(self):
        self._cache_groups = dict()
        self._diff_running = False

        for (_, m) in inspect.getmembers(type(self),
                                         predicate=lambda p:
                                         (inspect.ismethod or
                                         inspect.isdatadescriptor)):

            if hasattr(m, 'fget'):
                f = m.fget
            elif hasattr(m, 'im_func'):
                f = m.im_func
            else:
                continue

            fv = f.func_code.co_freevars

            try:
                closure = f.func_closure
            except AttributeError:
                continue

            if closure is None:
                continue

            vs = dict(zip(fv, (c.cell_contents for c in closure)))

            # this is used to make sure we are in the right function
            # i'm not proud of that, by the way
            if '_cache_identifier_pj97YCjgnp' not in vs:
                continue

            try:
                groups = vs['groups']
                method_name = re.match('^_cache_(.+)$',
                                       vs['cache_var_name']).group(1)
            except KeyError:
                continue

            for g in groups:
                if g not in self._cache_groups:
                    self._cache_groups[g] = []
                self._cache_groups[g].append(method_name)

            setattr(self, '_cache_' + method_name, None)
            setattr(self, '_cached_' + method_name, False)
            setattr(self, '_cached_args_' + method_name, dict())

        self._reset_profiler()

    def _reset_profiler(self):
        self.time = {}
        self.time_in = {}
        self.counter = {}
        cmethods = self._registered_methods()
        for m in cmethods:
            self.time[m] = 0
            self.time_in[m] = 0
            self.counter[m] = 0

    def diff(self, func, *args, **kwargs):
        if self._diff_running:
            return

        self._diff_running = True

        pp = pprint.PrettyPrinter(indent=4)

        print '*** Cache diff ***'

        self._print_groups()

        print '-- Cached methods --'
        cmethods = self._registered_methods()
        pp.pprint(cmethods)

        caches_before = self._get_caches()
        func(*args, **kwargs)
        caches_after = self._get_caches()

        print '-- Difference between caches -- '
        for cm in cmethods:

            try:
                changed = caches_before[cm] is not caches_after[cm]
            except:
                import ipdb; ipdb.set_trace()

            print cm+':', changed

            #if caches_before[cm] is not caches_after[cm]:
            #    print '%s: %s --> %s' % (cm, std_obj_repr(caches_before[cm]),
            #                                 std_obj_repr(caches_after[cm]))
        print '*** End ***'
        self._diff_running = False

    def _get_caches(self):
        methods = self._registered_methods()
        caches = dict()
        for m in methods:
           caches[m] = getattr(self, '_cache_' + m)
        return caches

    def _get_cached_methods(self):
        methods = self._registered_methods()
        cmethods = []
        for m in methods:
            # if hasattr(self, '_cache_' + m) and getattr(self, '_cached_' + m):
            cmethods.append(m)
        return cmethods

    def _registered_methods(self):
        methods = set()
        for g in self._cache_groups.values():
            for m in g:
                methods.add(m)
        return list(methods)

    def _print_groups(self):
        print '-- Cache groups and the respective registered methods --'
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(self._cache_groups)

    def clear_cache(self, *cache_groups):
        assert hasattr(self, '_cache_groups'), "Cache system failed because "\
                                   + "you did not call Cached's constructor."

        for g in cache_groups:
            if g not in self._cache_groups:
                raise Exception('Cache group %s does not exist.' % g)

            for method_name in self._cache_groups[g]:
                setattr(self, '_cache_' + method_name, None)
                setattr(self, '_cached_' + method_name, False)
                setattr(self, '_cached_args_' + method_name, dict())

    def fill_cache(self, method_name, value):
        if method_name not in self._registered_methods():
           raise ValueError("The method %s is not registered in this cache."
                            % method_name)
        setattr(self, '_cache_' + method_name, value)
        setattr(self, '_cached_' + method_name, True)

    def get_time_profile(self, reduced=True):
        if reduced:
            # reduce time to fields taht have been caculated at least once
            rv = {k:v for k,v in self.time.items() if self.counter[k]>0}
        else:
            rv = self.time
        return rv

    def get_timein_profile(self, reduced=True):
        if reduced:
            # reduce time to fields taht have been caculated at least once
            rv = {k:v for k,v in self.time_in.items() if self.counter[k]>0}
        else:
            rv = self.intime
        return rv

    def get_counter_profile(self, reduced=True):
        if reduced:
            # reduce time to fields taht have been caculated at least once
            rv = {k:v for k,v in self.counter.items() if v>0}
        else:
            rv = self.counter
        return rv

    def _profile(self, show=False, fs=10, rot=90, reduced=True):
        import pylab as pl
        import scipy as sp
        t_dict = self.get_time_profile(reduced=reduced)
        c_dict = self.get_counter_profile(reduced=reduced)
        labels = t_dict.keys()
        t = sp.array([t_dict[m] for m in labels])
        c = sp.array([c_dict[m] for m in labels])
        x = sp.arange(t.shape[0])
        plt = pl.subplot(211)
        pl.bar(x,t)
        plt.set_xticks(x + 0.5)
        #plt.set_xticklabels(labels, fontsize=fs, rotation=rot)
        plt = pl.subplot(212)
        pl.bar(x,c)
        plt.set_xticks(x + 0.5)
        plt.set_xticklabels(labels, fontsize=fs, rotation=rot)
        if show:
            pl.tight_layout()
            pl.show()


# This decorator works both with arguments or without
# (i.e., @cached, @cached("group_name"), @cached(["group_name_A",
#       "group_name_B"]), or @cached(exclude=['param1', 'param2', ...]))
def cached(*args, **kwargs):

    deco_without_arg = len(args) == 1 and inspect.isfunction(args[0])

    # cached_args stores the argument names and argument values with which
    # the method was called in
    # cached_args = dict()
    filter_out = []
    _groups = ['default']
    # This is a hacky thing so that one can know that a method_wrapper
    # object, defined bellow, really came from here.
    _cache_identifier_pj97YCjgnp = [False]

    if not deco_without_arg:
        if len(args) > 0:
            if type(args[0]) is list or type(args[0]) is tuple:
                _groups += list(args[0])
            else:
                _groups.append(args[0])

        if len(kwargs) == 1:
            assert 'exclude' in kwargs, ("'exclude' is the only keyword "
                                         "allowed here.")
            filter_out = kwargs['exclude']
    else:
        _groups.append(args[0].__name__)

    def real_cached(method):

        # dont look at this miserable hacky variable
        _cache_identifier_pj97YCjgnp[0] = True

        cache_var_name = '_cache_' + method.__name__
        valid_var_name = '_cached_' + method.__name__
        cached_args_name = '_cached_args_' + method.__name__
        groups = _groups

        debug = int(os.getenv('LIMIX_DEBUG', 0))
        if debug:
            return method

        def method_wrapper(self, *args, **kwargs):

            t0 = time()

            # dont look at this miserable hacky variable
            _cache_identifier_pj97YCjgnp[0] = True
            # this is meant to insert groups in this scope
            groups

            (argnames, argvalues) = _fetch_argnames_argvalues(method, args,
                                                              kwargs)

            t = zip(argnames, argvalues)
            provided_args = dict((x, y) for x, y in t)

            for f in filter_out:
                del provided_args[f]

            if getattr(self, valid_var_name, False) is False or\
                    provided_args != getattr(self, cached_args_name, dict()):
                t0in = time()
                result = method(self, *args, **kwargs)
                try:
                    self.time_in[method.__name__] += time() - t0in
                except:
                    import ipdb; ipdb.set_trace()
                setattr(self, cache_var_name, result)
                setattr(self, valid_var_name, True)
                if hasattr(self, cached_args_name):
                    getattr(self, cached_args_name).clear()
                    getattr(self, cached_args_name).update(provided_args)
                else:
                    setattr(self, cached_args_name, dict())
                self.counter[method.__name__] += 1.

            self.time[method.__name__] += time() - t0

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
    # argnames = inspect.getargspec(method)[0]
    nargs = method.func_code.co_argcount
    names = method.func_code.co_varnames
    argnames = list(names[:nargs])

    if len(argnames) == 1:
        return ([],[])

    # assert argnames[0] == 'self'
    # del argnames[argnames.index('self')]
    del argnames[0]

    # defaults = inspect.getargspec(method)[3]
    defaults = method.func_defaults
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
