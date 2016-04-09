from __future__ import print_function
import os
from os.path import join
import sys
import importlib

PKG_NAME = 'limix'
VERSION  = '0.7.18'

WORKDIR = os.path.abspath(os.path.dirname(__file__))

def ERR(text, bold=False):
    text = '\033[91m' + text + '\033[0m'
    if bold:
        text = '\033[1m' + text
    return text

def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = raw_input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")

def rreplace(s, old, new, occurrence):
    """ Replace last occurrence.
    """
    li = s.rsplit(old, occurrence)
    return new.join(li)

def try_import(pkgs):
    failed = []
    for pkg in pkgs:
        try:
            importlib.import_module(pkg)
        except ImportError:
            failed.append(pkg)
    return failed

def check_essentials():
    pkgs_failed = try_import(["setuptools", "cython", "numpy"])
    if len(pkgs_failed) > 0:

        plural = 's' if len(pkgs_failed) > 1 else ''
        pkgs_failed = [ERR(pf, True) for pf in pkgs_failed]

        s = rreplace(ERR(", ").join(pkgs_failed), ERR(", "), ERR(" and "), 1)

        msg  = ERR("Ops! We need the ")
        msg += s
        msg += ERR(" Python package{}".format(plural))
        msg += ERR(" installed before proceeding ")
        msg += ERR("with the {} installation.".format(PKG_NAME.upper()))
        print(msg)
        sys.exit(1)

check_essentials()

from setuptools import find_packages
from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext

def _check_gcc_cpp11(cc_name):
    import subprocess
    try:
        cmd = cc_name + ' -E -dM -std=c++11 -x c++ /dev/null > /dev/null'
        subprocess.check_call(cmd, shell=True)
    except subprocess.CalledProcessError:
        return False
    return True

class build_ext_subclass(build_ext):
    def build_extensions(self):
        if len(self.compiler.compiler) > 0:
            cc_name = self.compiler.compiler[0]
            stdcpp = '-std=c++11'
            if 'gcc' in cc_name and not _check_gcc_cpp11(cc_name):
                stdcpp = '-std=c++0x'
            for e in self.extensions:
                e.extra_compile_args.append(stdcpp)
        build_ext.build_extensions(self)

def globr(root, pattern):
    import fnmatch

    matches = []
    for root, _, filenames in os.walk(root):
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

def nlopt_files():
    src = open(join(WORKDIR, 'External', 'nlopt_src.files')).readlines()
    src = [join(WORKDIR, 'External', 'nlopt', s).strip() for s in src]
    hdr = globr(join(WORKDIR, 'External', 'nlopt'), '*/*.h')
    return (src, hdr)

def swig_opts():
    return ['-c++', '-outdir', join(WORKDIR, 'limix', 'deprecated'),
            '-I'+join(WORKDIR, 'src')]

def extra_compile_args():
    return ['-Wno-comment', '-Wno-unused-but-set-variable',
            '-Wno-overloaded-virtual', '-Wno-uninitialized',
            '-Wno-delete-non-virtual-dtor', '-Wunused-variable']

def extra_link_args():
    return []

def core_extension(reswig):
    import numpy as np

    (src, hdr) = nlopt_files()
    src.extend(globr(join(WORKDIR, 'src', 'limix'), '*.cpp'))
    hdr.extend(globr(join(WORKDIR, 'src', 'limix'), '*.h'))

    incl = ['src', 'External', join('External', 'nlopt')]
    incl = [join(WORKDIR, i) for i in incl]
    folder = join(WORKDIR, 'External', 'nlopt')
    incl += [join(folder, f) for f in os.listdir(folder)]
    incl = [i for i in incl if os.path.isdir(i)]
    incl.extend([np.get_include()])

    wrap_file = join(WORKDIR, 'src', 'interfaces', 'python', 'limix_wrap.cpp')
    i_file = join(WORKDIR, 'src', 'interfaces', 'python', 'limix.i')

    if os.path.exists(wrap_file):
        src.append(wrap_file)
    else:
        src.append(i_file)

    depends = src + hdr

    ext = Extension('limix.deprecated._core', src,
                    include_dirs=incl,
                    extra_compile_args=extra_compile_args(),
                    extra_link_args=extra_link_args(),
                    swig_opts=swig_opts(),
                    depends=depends)

    return ext

def ensemble_extension():
    import numpy as np

    src = [join(WORKDIR, 'cython', 'lmm_forest', 'SplittingCore.pyx')]
    incl = [join(WORKDIR, 'External'), np.get_include()]
    depends = src
    ext = Extension('limix.ensemble.SplittingCore', src,
                    language='c++',
                    include_dirs=incl,
                    extra_compile_args=extra_compile_args(),
                    extra_link_args=extra_link_args(),
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

def setup_package(reswig, yes):
    if sys.platform == 'darwin':
        mac_workaround()

    install_requires = ["sklearn", "pandas"]
    setup_requires = []

    # These are problematic packages (i.e., C/Fortran dependencies) to
    # install from pypi so we leave the option to the user for doing so.
    problematic_pkgs = ["scipy", "h5py", "matplotlib"]

    pkgs_failed = try_import(problematic_pkgs)
    if len(pkgs_failed) > 0:
        pkgs_failed = [ERR(pf, True) for pf in pkgs_failed]
        fpkgs = rreplace(ERR(", ").join(pkgs_failed), ERR(", "),
                         ERR(" and "), 1)
        plural = 's' if len(pkgs_failed) > 1 else ''
        msg  = ERR("The ") + fpkgs + ERR(" Python package%s " % plural)
        plural = 'are' if len(pkgs_failed) > 1 else 'is'
        msg += ERR("%s needed by " % plural)
        msg += ERR(PKG_NAME.upper() + " but could not be found in your ")
        msg += ERR("system.")
        print(msg)
        print("We recommend their installation to be done through "
              "package managers (e.g., conda, canopy, apt-get, yum, brew) "
              "but we can also try to install them right now.")

        if not yes:
            yes = query_yes_no("Do you want to try to install them right now?",
                               "no")
        if yes:
            install_requires.extend(problematic_pkgs)
        else:
            print("Good choice! Their installation tend to be painless via "
                  "package managers.")
            sys.exit(0)

    src_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    old_path = os.getcwd()
    os.chdir(src_path)
    sys.path.insert(0, src_path)

    write_version()

    metadata = dict(
        name=PKG_NAME,
        description="A flexible and fast mixed model "+
                    "toolbox written in C++/python",
        long_description=open(join(WORKDIR, 'README'), 'r').read(),
        keywords='linear mixed models, GWAS, QTL, ' +
                 'Variance component modelling',
        maintainer="Limix Developers",
        author="Christoph Lippert, Paolo Casale, Oliver Stegle",
        author_email="stegle@ebi.ac.uk",
        maintainer_email="stegle@ebi.ac.uk",
        version=VERSION,
        test_suite='setup.get_test_suite',
        packages=find_packages(exclude=['tests', 'test', 'test_limix*']),
        license="BSD",
        url='http://pmbio.github.io/limix/',
        install_requires=install_requires,
        setup_requires=setup_requires,
        zip_safe=False,
        ext_modules=[core_extension(reswig)] + ensemble_extension(),
        cmdclass=dict(build_ext=build_ext_subclass),
        entry_points={
            'console_scripts':[
                'limix_runner=limix.scripts.limix_runner:entry_point',
                'mtSet_postprocess=limix.scripts.mtSet_postprocess:entry_point',
                'mtSet_preprocess=limix.scripts.mtSet_preprocess:entry_point',
                'mtSet_simPheno=limix.scripts.mtSet_simPheno:entry_point',
                'mtSet_analyze=limix.scripts.mtSet_analyze:entry_point',
                'limix_converter=limix.scripts.limix_converter:entry_point'
            ]
        }
    )

    try:
        from distutils.command.bdist_conda import CondaDistribution
        metadata['distclass'] = CondaDistribution
        metadata['conda_buildnum'] = 1
        metadata['conda_features'] = ['mkl']
    except ImportError:
        pass

    # http://stackoverflow.com/a/29634231
    import distutils.sysconfig
    cfg_vars = distutils.sysconfig.get_config_vars()
    for key, value in cfg_vars.items():
        if type(value) == str:
            cfg_vars[key] = value.replace("-Wstrict-prototypes", "")

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

    yes = False
    if "--yes" in sys.argv:
        yes = True
        sys.argv.remove("--yes")
    setup_package(reswig, yes)
