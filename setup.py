#change CC to include -std=c++0x flags.
#this is a hack as distutils does not permit specifying seperate build flags for .c and .cpp files
import os
#os.environ['CC'] = 'gcc -std=c++0x'

import distutils.cmd
from setuptools import find_packages
import sys,os,re
from distutils.core import Extension
from distutils.command.build import build

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

import pdb
import glob

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
    numpy_inc_path = [numpy.get_include()]
    include_dirs.extend(numpy_inc_path)
    return include_dirs

def get_swig_opts():
    swig_opts=['-c++', '-Isrc','-outdir','src/interfaces/python/limix']
    return swig_opts

def get_extra_compile_args():
    return ['-std=c++0x']
    #return []

import numpy

class CustomBuild(build):
    sub_commands = [
        ('build_ext', build.has_ext_modules), 
        ('build_py', build.has_pure_modules),
        ('build_clib', build.has_c_libraries), 
        ('build_scripts', build.has_scripts),
    ]

reswig = False
if '--reswig' in sys.argv:
    index = sys.argv.index('--reswig')
    sys.argv.pop(index)  # Removes the '--foo'
    reswig = True

packages = ['limix', 'limix.io', 'limix.modules', 'limix.stats', 'limix.utils']
reqs = ['numpy', 'scipy', 'pygp >=1.1.07', 'matplotlib >=1.2']

FL = get_source_files(reswig=reswig)

#create setup:
setup(
    name = 'limix',
    version = '0.7.3',
    cmdclass={'build': CustomBuild},
    author = 'Christoph Lippert, Paolo Casale, Oliver Stegle',
    author_email = "stegle@ebi.ac.uk",
    description = ('A flexible and fast mixed model toolbox written in C++/python'),
    url = "http://",
    long_description = read('README'),
    license = 'BSD',
    keywords = 'linear mixed models, GWAS, QTL, Variance component modelling',
    ext_package = 'limix',
    ext_modules = [Extension('_core',get_source_files(reswig=reswig),include_dirs=get_include_dirs(),swig_opts=get_swig_opts(),extra_compile_args = get_extra_compile_args())],
    py_modules = ['limix.core'],
    scripts = ['src/interfaces/python/bin/limix_runner'],
    #packages = find_packages(),
    packages = packages,
    package_dir = {'limix': 'src/interfaces/python/limix'},
    #use manual build system building on scons
    #cmdclass = {'build_py': build_py_cmd},
    #dependencies
    #requires = ['scipy','numpy','matplotlib','pandas','scons'],
    requires=map(lambda x: x.split(" ")[0], reqs),
    install_requires = reqs
    )
