import distutils.cmd
import sys,os,re


try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

import pdb
import glob

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

__revision__ = "src/script/scons.py  2013/03/03 09:48:35 garyo"

__version__ = "2.3.0"

__build__ = ""

__buildsys__ = "reepicheep"

__date__ = "2013/03/03 09:48:35"

__developer__ = "garyo"
##############################################################################
# BEGIN STANDARD SCons SCRIPT HEADER
#
# This is the cut-and-paste logic so that a self-contained script can
# interoperate correctly with different SCons versions and installation
# locations for the engine.  If you modify anything in this section, you
# should also change other scripts that use this same header.
##############################################################################

# Strip the script directory from sys.path() so on case-insensitive
# (WIN32) systems Python doesn't think that the "scons" script is the
# "SCons" package.  Replace it with our own library directories
# (version-specific first, in case they installed by hand there,
# followed by generic) so we pick up the right version of the build
# engine modules if they're in either directory.


if sys.version_info >= (3,0,0):
    msg = "scons: *** SCons version %s does not run under Python version %s.\n\
Python 3 is not yet supported.\n"
    sys.stderr.write(msg % (__version__, sys.version.split()[0]))
    sys.exit(1)


script_dir = sys.path[0]

if script_dir in sys.path:
    sys.path.remove(script_dir)

libs = []

if "SCONS_LIB_DIR" in os.environ:
    libs.append(os.environ["SCONS_LIB_DIR"])

local_version = 'scons-local-' + __version__
local = 'scons-local'
if script_dir:
    local_version = os.path.join(script_dir, local_version)
    local = os.path.join(script_dir, local)
libs.append(os.path.abspath(local_version))
libs.append(os.path.abspath(local))

scons_version = 'scons-%s' % __version__

# preferred order of scons lookup paths
prefs = []

try:
    import pkg_resources
except ImportError:
    pass
else:
    # when running from an egg add the egg's directory 
    try:
        d = pkg_resources.get_distribution('scons')
    except pkg_resources.DistributionNotFound:
        pass
    else:
        prefs.append(d.location)

if sys.platform == 'win32':
    # sys.prefix is (likely) C:\Python*;
    # check only C:\Python*.
    prefs.append(sys.prefix)
    prefs.append(os.path.join(sys.prefix, 'Lib', 'site-packages'))
else:
    # On other (POSIX) platforms, things are more complicated due to
    # the variety of path names and library locations.  Try to be smart
    # about it.
    if script_dir == 'bin':
        # script_dir is `pwd`/bin;
        # check `pwd`/lib/scons*.
        prefs.append(os.getcwd())
    else:
        if script_dir == '.' or script_dir == '':
            script_dir = os.getcwd()
        head, tail = os.path.split(script_dir)
        if tail == "bin":
            # script_dir is /foo/bin;
            # check /foo/lib/scons*.
            prefs.append(head)

    head, tail = os.path.split(sys.prefix)
    if tail == "usr":
        # sys.prefix is /foo/usr;
        # check /foo/usr/lib/scons* first,
        # then /foo/usr/local/lib/scons*.
        prefs.append(sys.prefix)
        prefs.append(os.path.join(sys.prefix, "local"))
    elif tail == "local":
        h, t = os.path.split(head)
        if t == "usr":
            # sys.prefix is /foo/usr/local;
            # check /foo/usr/local/lib/scons* first,
            # then /foo/usr/lib/scons*.
            prefs.append(sys.prefix)
            prefs.append(head)
        else:
            # sys.prefix is /foo/local;
            # check only /foo/local/lib/scons*.
            prefs.append(sys.prefix)
    else:
        # sys.prefix is /foo (ends in neither /usr or /local);
        # check only /foo/lib/scons*.
        prefs.append(sys.prefix)

    temp = [os.path.join(x, 'lib') for x in prefs]
    temp.extend([os.path.join(x,
                                           'lib',
                                           'python' + sys.version[:3],
                                           'site-packages') for x in prefs])
    prefs = temp

    # Add the parent directory of the current python's library to the
    # preferences.  On SuSE-91/AMD64, for example, this is /usr/lib64,
    # not /usr/lib.
    try:
        libpath = os.__file__
    except AttributeError:
        pass
    else:
        # Split /usr/libfoo/python*/os.py to /usr/libfoo/python*.
        libpath, tail = os.path.split(libpath)
        # Split /usr/libfoo/python* to /usr/libfoo
        libpath, tail = os.path.split(libpath)
        # Check /usr/libfoo/scons*.
        prefs.append(libpath)

# Look first for 'scons-__version__' in all of our preference libs,
# then for 'scons'.
libs.extend([os.path.join(x, scons_version) for x in prefs])
libs.extend([os.path.join(x, 'scons') for x in prefs])

sys.path = libs + sys.path

##############################################################################
# END STANDARD SCons SCRIPT HEADER
##############################################################################


def file_list_recursive(dir_name,exclude_list=[]):
    """create a recursive file list"""
    FL = []
    for root, dirs, files in os.walk(dir_name):
        FL_ = [os.path.join(root,fn) for fn in files]
        #filter and append
        for fn in FL_:
            if not any([ex in fn for ex in exclude_list]):
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
    try:
        import SCons
        sconsversion = SCons.__version__
    except ImportError:
        raise ImportError("LIMIX requires scons")

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
    try:
        assert StrictVersion(strip_rc(sconsversion)) >= min_versions['scons']
    except AssertionError:
        raise ImportError("scons version is %s. Requires >= %s" %
                (sconsversion, min_versions['scons']))


class build_py_cmd(distutils.cmd.Command):
    def initialize_options(self):
        pass

    def finalize_options(self):
        pass
    def run(self):
        print("Calling SCons to build the module")
        SCons.Script.main()
        pbs.scons()
    def get_source_files(self):
        FL = []
        FL.extend(file_list_recursive('./src',exclude_list=['src/archive','src/testing']))
        FL.extend(file_list_recursive('./External'))
        FL.extend(file_list_recursive('./tests'))
        FL.extend(file_list_recursive('./site_scons'))
        FL.extend(file_list_recursive('./doc/tutorials'))
        FL.extend(['SConstruct','README','license.txt','doc/doxy.cfg'])
        return FL
    #data files
    #data_files=[('', ['license.txt'])]
    data_files= []


if __name__ == '__main__':
    min_versions = {
        'numpy' : '1.6.0',
        'scipy' : '0.9.0',
        'pandas' : '0.12.0',
        'scons' : '2.1.0',
                   }
    check_versions(min_versions)

    import SCons.Script
    setup(
        name = 'limix',
        version = '0.6.5',
        author = 'Christoph Lippert, Paolo Casale, Oliver Stegle',
        author_email = "lippert@microsoft.com, stegle@ebi.ac.uk",
        description = ('A flexible and fast mixed model toolbox written in C++/python'),
        url = "http://",
        long_description = read('README'),
        license = 'BSD',
        keywords = 'linear mixed models, GWAS, QTL',
        scripts = ['src/interfaces/python/bin/limix_runner.py'],
        packages = ['limix'],
        #package_dir = {'': 'build/release'},
        #use manual build system building on scons
        cmdclass = {'build_py': build_py_cmd},
        #dependencies
        #requires = ['scipy','numpy','matplotlib','pandas','scons'],
        requires = ['scipy','numpy','matplotlib','pandas'],
        install_requires = ['scons>=2.3.0']
        )
