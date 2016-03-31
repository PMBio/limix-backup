from __future__ import division, print_function
import os
from os.path import join
import sys
import importlib

PKG_NAME = 'limix'
VERSION  = '0.7.12'

def try_import(pkg):
    try:
        importlib.import_module(pkg)
    except ImportError:
        print("Error: %s package couldn't be found." % pkg +
              " Please, install it so I can proceed.")
        sys.exit(1)

# These are problematic packages (i.e., C/Fortran dependencies) to
# install from pypi so we leave the option to the user for doing so.
try_import("numpy")
try_import("scipy")
try_import("cython")
try_import("h5py")
try_import("pandas")
try_import("sklearn")
try_import("matplotlib")

from setuptools import find_packages
from setuptools import setup
from setuptools.extension import Extension
import numpy as np
from Cython.Build import cythonize
from Cython.Distutils import build_ext

def globr(root, pattern):
    import fnmatch
    import os

    matches = []
    for root, dirnames, filenames in os.walk(root):
        for filename in fnmatch.filter(filenames, pattern):
            matches.append(os.path.join(root, filename))

    return matches

def mac_workaround():
    import platform
    from distutils import sysconfig

    conf_vars = sysconfig.get_config_vars()
    vers = platform.mac_ver()[0].split('.')
    if len(vers) == 3:
        conf_vars['MACOSX_DEPLOYMENT_TARGET'] =\
            vers[0] + '.' + vers[1]
    else:
        conf_vars['MACOSX_DEPLOYMENT_TARGET'] = platform.mac_ver()[0]

_curdir = os.path.abspath(os.path.dirname(__file__))

try:
    from distutils.command.bdist_conda import CondaDistribution
except ImportError:
    conda_present = False
else:
    conda_present = True

def nlopt_files():
    src = open(join(_curdir, 'External', 'nlopt_src.files')).readlines()
    src = [join(_curdir, 'External', 'nlopt', s).strip() for s in src]
    hdr = globr(join(_curdir, 'External', 'nlopt'), '*/*.h')
    return (src, hdr)

def swig_opts():
    return ['-c++', '-outdir', join(_curdir, 'limix', 'deprecated'),
            '-I'+join(_curdir, 'src')]

def extra_compile_args():
    return ['-Wno-comment', '-Wno-unused-but-set-variable',
            '-Wno-overloaded-virtual', '-Wno-uninitialized',
            '-Wno-unused-const-variable', '-Wno-unknown-warning-option',
            '-Wno-shorten-64-to-32']

def core_extension(reswig):
    (src, hdr) = nlopt_files()
    src.extend(globr(join(_curdir, 'src', 'limix'), '*.cpp'))
    hdr.extend(globr(join(_curdir, 'src', 'limix'), '*.h'))

    incl = ['src', 'External', join('External', 'nlopt')]
    incl = [join(_curdir, i) for i in incl]
    folder = join(_curdir, 'External', 'nlopt')
    incl += [join(folder, f) for f in os.listdir(folder)]
    incl = [i for i in incl if os.path.isdir(i)]
    incl.extend([np.get_include()])

    src.append(join(_curdir, 'src', 'interfaces', 'python', 'limix.i'))

    depends = src + hdr

    ext = Extension('limix.deprecated._core', src,
                    include_dirs=incl,
                    extra_compile_args=extra_compile_args(),
                    swig_opts=swig_opts(),
                    depends=depends)

    return ext

def ensemble_extension():
    src = [join(_curdir, 'cython', 'lmm_forest', 'SplittingCore.pyx')]
    incl = [join(_curdir, 'External'), np.get_include()]
    depends = src
    ext = Extension('limix.ensemble.SplittingCore', src,
                    language='c++',
                    include_dirs=incl,
                    extra_compile_args=extra_compile_args(),
                    depends=depends)
    return cythonize(ext)

def write_version():
    cnt = """
# THIS FILE IS GENERATED FROM %(package_name)s SETUP.PY
version = '%(version)s'
"""
    filename = os.path.join(PKG_NAME, 'version.py')
    a = open(filename, 'w')
    try:
        a.write(cnt % {'version': VERSION,
                       'package_name': PKG_NAME.upper()})
    finally:
        a.close()

def get_test_suite():
    from unittest import TestLoader
    return TestLoader().discover(PKG_NAME)

def setup_package(reswig):
    if sys.platform == 'darwin':
        mac_workaround()

    src_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    old_path = os.getcwd()
    os.chdir(src_path)
    sys.path.insert(0, src_path)

    write_version()

    install_requires = []
    setup_requires = []

    metadata = dict(
        name=PKG_NAME,
        description="A flexible and fast mixed model "+
                    "toolbox written in C++/python",
        long_description=open(join(_curdir, 'README'), 'r').read(),
        keywords='linear mixed models, GWAS, QTL, ' +
                 'Variance component modelling',
        maintainer="Limix Developers",
        author="Christoph Lippert, Paolo Casale, Oliver Stegle",
        author_email="stegle@ebi.ac.uk",
        maintainer_email="stegle@ebi.ac.uk",
        version=VERSION,
        test_suite='setup.get_test_suite',
        packages=find_packages(exclude=['tests', 'test', 'test_limix*',
                                        'limix.modules2*']),
        license="BSD",
        url='http://pmbio.github.io/limix/',
        install_requires=install_requires,
        setup_requires=setup_requires,
        zip_safe=False,
        ext_modules=[core_extension(reswig)] + ensemble_extension(),
        cmdclass=dict(build_ext=build_ext),
        entry_points={
            'console_scripts': [
                'limix_runner=limix.scripts.limix_runner:entry_point',
                'mtSet_postprocess=limix.scripts.mtSet_postprocess:entry_point',
                'mtSet_preprocess=limix.scripts.mtSet_preprocess:entry_point',
                'mtSet_simPheno=limix.scripts.mtSet_simPheno:entry_point',
                'mtSet_analyze=limix.scripts.mtSet_analyze:entry_point',
                'limix_converter=limix.scripts.limix_converter:entry_point',

            ]
        }
    )

    if conda_present:
        metadata['distclass'] = CondaDistribution
        metadata['conda_buildnum'] = 1
        metadata['conda_features'] = ['mkl']

    try:
        setup(**metadata)
    finally:
        del sys.path[0]
        os.chdir(old_path)

if __name__ == '__main__':
    reswig = False
    if "--reswig" in sys.argv:
        reswig = True
        sys.argv.remove("--reswig")
    setup_package(reswig)
