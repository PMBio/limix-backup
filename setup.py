#change CC to include -std=c++0x flags.
#this is a hack as distutils does not permit specifying seperate build flags for .c and .cpp files
import os
#os.environ['CC'] = 'gcc -std=c++0x'

import distutils.cmd
from setuptools import find_packages
import sys,os,re
from distutils.core import Extension
from distutils.command.build import build
from distutils.command.build_ext import build_ext
from distutils.errors import CompileError
# from redirect import stdout_redirected, merged_stderr_stdout


# try:
from setuptools import setup
# except ImportError:
    # from distutils.core import setup

import pdb
import glob








###################### THIS SHOULD BE IN A MODULE ######################
import os
import sys
from contextlib import contextmanager

def fileno(file_or_fd):
    fd = getattr(file_or_fd, 'fileno', lambda: file_or_fd)()
    if not isinstance(fd, int):
        raise ValueError("Expected a file (`.fileno()`) or a file descriptor")
    return fd

@contextmanager
def stdout_redirected(to=os.devnull, stdout=None):
    if stdout is None:
       stdout = sys.stdout

    stdout_fd = fileno(stdout)
    # copy stdout_fd before it is overwritten
    #NOTE: `copied` is inheritable on Windows when duplicating a standard stream
    with os.fdopen(os.dup(stdout_fd), 'wb') as copied:
        stdout.flush()  # flush library buffers that dup2 knows nothing about
        try:
            os.dup2(fileno(to), stdout_fd)  # $ exec >&to
        except ValueError:  # filename
            with open(to, 'wb') as to_file:
                os.dup2(to_file.fileno(), stdout_fd)  # $ exec > to
        try:
            yield stdout # allow code to be run with the redirected stdout
        finally:
            # restore stdout to its previous value
            #NOTE: dup2 makes stdout_fd inheritable unconditionally
            stdout.flush()
            os.dup2(copied.fileno(), stdout_fd)  # $ exec >&copied

def merged_stderr_stdout():  # $ exec 2>&1
    return stdout_redirected(to=sys.stdout, stdout=sys.stderr)
###################### THIS SHOULD BE IN A MODULE ######################




















import pip.commands.install
def install_distributions(distributions):
    command = pip.commands.install.InstallCommand()
    opts, args = command.parser.parse_args()
    # TBD, why do we have to run the next part here twice before actual install
    requirement_set = command.run(opts, distributions)
    requirement_set = command.run(opts, distributions)
    requirement_set.install(opts)
try:
    import numpy as np
except ImportError:
    install_distributions(['numpy'])
    import numpy as np
try:
    import Cython
except ImportError:
    install_distributions(['Cython'])

from Cython.Build import cythonize


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

def file_list_recursive(dir_name,exclude_list=[],ext=[]):
    """create a recursive file list"""
    FL = []
    for root, dirs, files in os.walk(dir_name):
        FL_ = [os.path.join(root,fn) for fn in files]
        #filter and append
        for fn in FL_:
            if not any([ex in fn for ex in exclude_list]):
                if (os.path.splitext(fn)[1] in ext):
                    FL.append(fn)
    return FL

def strip_rc(version):
    return re.sub(r"rc\d+$", "", version)

def check_versions(min_versions):
    """
    Check versions of dependency packages
    """
    from distutils.version import StrictVersion

    try:
        import scipy
        spversion = scipy.__version__
    except ImportError:
        raise ImportError("LIMIX requires scipy")

    try:
        import numpy
        npversion = numpy.__version__
    except ImportError:
        raise ImportError("LIMIX requires numpy")

    try:
        import pandas
        pandasversion = pandas.__version__
    except ImportError:
        raise ImportError("LIMIX requires pandas")
    #match version numbers
    try:
        assert StrictVersion(strip_rc(npversion)) >= min_versions['numpy']
    except AssertionError:
        raise ImportError("Numpy version is %s. Requires >= %s" %
                (npversion, min_versions['numpy']))
    try:
        assert StrictVersion(strip_rc(spversion)) >= min_versions['scipy']
    except AssertionError:
        raise ImportError("Scipy version is %s. Requires >= %s" %
                (spversion, min_versions['scipy']))
    try:
        assert StrictVersion(strip_rc(pandasversion)) >= min_versions['pandas']
    except AssertionError:
        raise ImportError("pandas version is %s. Requires >= %s" %
                (pandasversion, min_versions['pandas']))

def get_source_files(reswig=True):
    """build list of source files. swig=True means the interfaces is 'reswigged'. Otherwise the distribution
    version of the numpy python wrappers are retained"""
    FL = []
    #nlopt sources files
    nlopt=['direct/DIRect.cpp',
        'direct/direct_wrap.cpp',
        'direct/DIRserial.cpp',
        'direct/DIRsubrout.cpp',
        'cdirect/cdirect.cpp','cdirect/hybrid.cpp',
        'praxis/praxis.cpp','luksan/plis.cpp','luksan/plip.cpp','luksan/pnet.cpp', 'luksan/mssubs.cpp','luksan/pssubs.cpp',
        'crs/crs.cpp',
        'mlsl/mlsl.cpp',
        'mma/mma.cpp','mma/ccsa_quadratic.cpp',
        'cobyla/cobyla.cpp',
        'newuoa/newuoa.cpp',
        'neldermead/nldrmd.cpp','neldermead/sbplx.cpp',
        'auglag/auglag.cpp',
        'esch/esch.cpp',
        'bobyqa/bobyqa.cpp',
        'isres/isres.cpp',
        'slsqp/slsqp.cpp',
        'api/general.cpp','api/options.cpp','api/optimize.cpp','api/deprecated.cpp','api/f77api.cpp',
        'util/mt19937ar.cpp','util/sobolseq.cpp','util/timer.cpp','util/stop.cpp','util/redblack.cpp','util/qsort_r.cpp','util/rescale.cpp',
        'stogo/global.cc','stogo/linalg.cc','stogo/local.cc','stogo/stogo.cc','stogo/tools.cc'
        ]
    #limix sourcs files
    #python wrapper
    FL.extend(file_list_recursive('./src',exclude_list=['src/archive','src/testing','src/interfaces'],ext=['.cpp','.c']))
    #nlopt
    nlopt = ['./External/nlopt/%s' % fn for fn in nlopt]
    #add header files
    if reswig:
        FL.extend(['src/interfaces/python/limix.i'])
    else:
        pass
        FL.extend(['src/interfaces/python/limix_wrap.cpp'])
    FL.extend(nlopt)
    return FL

def get_include_dirs():
    include_dirs = ['src']
    include_dirs.extend(['External','External/nlopt'])
    nlopt_include_dir = ['stogo','util','direct','cdirect','praxis','luksan','crs','mlsl','mma','cobyla','newuoa','neldermead','auglag','bobyqa','isres','slsqp','api','esch']
    nlopt_include_dir = ['./External/nlopt/%s' % fn for fn in nlopt_include_dir]
    include_dirs.extend(nlopt_include_dir)
    #add numpy include dir
    numpy_inc_path = [np.get_include()]
    include_dirs.extend(numpy_inc_path)
    return include_dirs

def get_swig_opts():
    swig_opts=['-c++', '-Isrc','-outdir','limix/deprecated']
    return swig_opts

def get_extra_compile_args():
    return ['-Wno-comment', '-Wno-unused-but-set-variable',
            '-Wno-overloaded-virtual', '-Wno-uninitialized',
            '-Wno-unused-const-variable', '-Wno-unknown-warning-option',
            '-Wno-shorten-64-to-32']

def try_to_add_compile_args():
    return ['-std=c++0x', '-stdlib=libc++']

class CustomBuild(build):
    sub_commands = [
        ('build_ext', build.has_ext_modules),
        ('build_py', build.has_pure_modules),
        ('build_clib', build.has_c_libraries),
        ('build_scripts', build.has_scripts),
    ]

class CustomBuildExt(build_ext):
    def build_extensions(self):
        import tempfile

        flags = try_to_add_compile_args()

        f = tempfile.NamedTemporaryFile(suffix=".cpp", delete=True)
        f.name
        c = self.compiler

        ok_flags = []

        with stdout_redirected(), merged_stderr_stdout():
            for flag in flags:
                try:
                    c.compile([f.name], extra_postargs=ok_flags+[flag])
                except CompileError:
                    pass
                else:
                    ok_flags.append(flag)

        for ext in self.extensions:
            ext.extra_compile_args += ok_flags

        f.close()
        build_ext.build_extensions(self)

reswig = False
if '--reswig' in sys.argv:
    index = sys.argv.index('--reswig')
    sys.argv.pop(index)  # Removes the '--foo'
    reswig = True

#1. find packages (parses the local 'limix' tree')
# exclude limix.deprecated. This is a placeholder and will be replaced with the
# actual deprecated limix source tree
#packages = find_packages(exclude=['limix.deprecated'])
packages = find_packages(exclude=['tests', 'test', 'test_limix*',
                                  'limix.modules2*'])
#3. add depcreated limix packages in src/interfaces/python (see below)
#packages.extend(['limix.deprecated', 'limix.deprecated.io',
#                 'limix.deprecated.modules', 'limix.deprecated.stats',
#                 'limix.deprecated.utils'])
# reqs = ['scikit-learn', 'h5py', 'numpy', 'scipy', 'matplotlib']
reqs = ['Cython', 'h5py']
# reqs = []

FL = get_source_files(reswig=reswig)

#fore deployment version to be the current MAC release and not what is stored in distutils
#this is key for some distributions like anaconda, which otherwise build for an outdated target.
from sys import platform as _platform
if _platform == 'darwin':
    from distutils import sysconfig
    import platform
    vers = platform.mac_ver()[0].split('.')
    # if vers[0] == '10' or vers[0] == '9':
    #     sysconfig._config_vars['MACOSX_DEPLOYMENT_TARGET'] = '10.8'
    # else:
    if len(vers) == 3:
        sysconfig._config_vars['MACOSX_DEPLOYMENT_TARGET'] =\
            vers[0] + '.' + vers[1]
    else:
        sysconfig._config_vars['MACOSX_DEPLOYMENT_TARGET'] = platform.mac_ver()[0]

def get_test_suite():
    from unittest import TestLoader
    from unittest import TestSuite
    test_suite1 = TestLoader().discover('limix')
    test_suite2 = TestLoader().discover('test_limix')
    return TestSuite([test_suite1, test_suite2])


extensions = [Extension('deprecated._core',
                         get_source_files(reswig=reswig),
                         include_dirs=get_include_dirs(),
                         swig_opts=get_swig_opts(),
                         extra_compile_args=get_extra_compile_args())]
extensions += cythonize(Extension(name="ensemble.SplittingCore",
                        language="c++",
                        sources=["cython/lmm_forest/SplittingCore.pyx"],
                        include_dirs=get_include_dirs() + ['.'],
                        extra_compile_args=get_extra_compile_args()))

import versioneer
cmdclass = versioneer.get_cmdclass()
cmdclass['build'] = CustomBuild
cmdclass['build_ext'] = CustomBuildExt
setup(
    name = 'limix',
    version = versioneer.get_version(),
    cmdclass = cmdclass,
    author = 'Christoph Lippert, Paolo Casale, Oliver Stegle',
    author_email = "stegle@ebi.ac.uk",
    description = ('A flexible and fast mixed model toolbox written in C++/python'),
    url = "http://",
    long_description = read('README'),
    license = 'BSD',
    keywords = 'linear mixed models, GWAS, QTL, Variance component modelling',
    ext_package = 'limix',
    ext_modules = extensions,
    py_modules = ['limix.deprecated.core'],
    scripts = ['scripts/limix_runner','scripts/mtSet_postprocess',
               'scripts/mtSet_preprocess','scripts/mtSet_simPheno',
               'scripts/mtSet_analyze', 'scripts/limix_converter'],
    packages = packages,
    package_dir = {'limix': 'limix'},
    # requires=reqs, seems that we dont need it, and it was causing trouble
    install_requires=reqs,
    setup_requires = ['numpy', 'Cython'],
    test_suite='setup.get_test_suite'
    )
