#use systems path for building commands, etc.
import os
import sys
import subprocess
import pdb
import distutils.sysconfig, os


#import autoconfig like handling
from ACGenerateFile import *

#build prefix
build_prefix = 'build'
#maximum number of build jobs
max_jobs = 2
#build tool
build_tool = 'default'
#on windows, we could also use mingw 
#rerun swig by default?
reswig_default = False 
static_bind_default = False

#windows currently only supports static bind
if(sys.platform =='win32'):
  static_bind_default =True

### 0. get default compiler settings from distutils
dist_vars = distutils.sysconfig.get_config_vars('CC', 'CXX', 'OPT', 'BASECFLAGS', 'CCSHARED', 'LDSHARED', 'SO')
for i in range(len(dist_vars)):
   if dist_vars[i] is None:
      dist_vars[i] = ""
(cc, cxx, opt, basecflags, ccshared, ldshared, so_ext) = dist_vars

### 1. Command line handling
#command line handling
#build mode
AddOption('--mode',dest='mode',
   type='string',nargs=1,action='store',default='release',help='Build mode: debug/release')

#support for vcflib?
AddOption('--with-zlib', dest='with_zlib', action='store_true',
help='Support for compressed files (zlib)', default=False)

#build python interface?
AddOption('--without-python', dest='with_python', action='store_false',
help='Disable python interface', default=True)

#run swig?
AddOption('--reswig', dest='reswig', action='store_true', help='Run swig?', default=reswig_default)

#build doxygen documentation?
AddOption('--with-documentation', dest='with_documentation', action='store_true',
help='Build doxygen documenation', default=False)

#run unit tests?
AddOption('--with-tests', dest='with_tests', action='store_true',
help='Run unit tests after build', default=False)

#run unit tests?
AddOption('--with-mkl', dest='with_mkl', action='store_true',
help='Bind against MKL performance librabries', default=False)

#build development tools for c++ (standalone snipsets)
AddOption('--with-developcpp', dest='with_developcpp', action='store_true',
help='Build development only commandline tools?', default=False)

#override  build settings
AddOption('--CXX',dest='CXX',type='string',nargs=1,action='store',default=cxx,help='Manual specified CXX')
#override CC compiler (for open MPI)
AddOption('--CC',dest='CC',type='string',nargs=1,action='store',default=cc,help='Manual specified CC')
#override  build tool
AddOption('--build_tool',dest='build_tool',type='string',nargs=1,action='store',default=build_tool,help='Manual specification of build tool')
#additional cxx flags
AddOption('--CXXflags',dest='CXXflags',type='string',nargs=1,action='store',default=None,help='additional cxx build flags')
#use static bind?
#note: if enabled, the library is effectively GPL as nlopt is lesser GPL and hence we require dynamic binding
AddOption('--static-bind',dest='static_bind',action='store_true',help='Enforce static binding of external libs',default=static_bind_default)

#### options that are needed for distutils compatibility ####
#set output?
AddOption('--record',dest='record',type='string',nargs=1,action='store',default='',help='Record build output for distutils build')

#dummpy options to please setup tools
AddOption('--compile', dest='compile', action='store_true', help='No action', default=False)
AddOption('--single-version-externally-managed', dest='single-version-externally-managed', action='store_true', help='No action', default=False)

# 2. parallel build options
# Do parallel builds by default
n_jobs = 1

try:
    import multiprocessing
    n_jobs = min(max_jobs,multiprocessing.cpu_count())
except:
    pass

SetOption('num_jobs', n_jobs)


#3. parse build options:
build_options= {}
build_options['with_zlib'] = GetOption('with_zlib')
build_options['with_mkl'] = GetOption('with_mkl')
build_options['with_python'] = GetOption('with_python')
build_options['with_developcpp'] = GetOption('with_developcpp')
build_options['with_tests'] = GetOption('with_tests')
build_options['with_documentation'] = GetOption('with_documentation')
build_options['CXX'] = GetOption('CXX')
build_options['CXXflags'] = GetOption('CXXflags')
build_options['CC'] = GetOption('CC')
build_options['build_tool'] = GetOption('build_tool')
build_options['reswig'] = GetOption('reswig')
build_options['record'] = GetOption('record')
build_options['static_bind'] = GetOption('static_bind')

### 2. build mode
#build mode:
mymode = GetOption('mode')
#check if the user has been naughty: only 'debug' or 'release' allowed
if not (mymode in ['debug', 'release']):
   print "Error: expected 'debug' or 'release', found: " + mymode
   Exit(1)

#tell the user what we're doing
#create build prefix
build_prefix = os.path.join(build_prefix,mymode + '.' + sys.platform)
print '**** Compiling in ' + build_prefix + ' mode...'
cflags = basecflags.split(' ')
cxxflags = []
linkflags = []
debugcflags   = ['-DDEBUG']   #extra compile flags for debug
releasecflags = opt.split(' ')         #extra compile flags for release
releasecflags.extend(['-DRELEASE'])         #extra compile flags for release
debuglinkflags = []
releaselinkflags = []

#libraries to be included in binaries or fully linked libraries (names)
limix_LIBS_str = ['nlopt','limix']
#library handles
limix_LIBS = []

#build environment
copy_env = ['PATH','INCLUDE','LIB','TMP','MKLROOT']
ENV = {}
for key in copy_env:
    if key in os.environ.keys():
       ENV[key] = os.environ[key]

#TOOL_SUST
env = Environment(SHLIBPREFIX="",ENV=ENV,tools = [build_options['build_tool'],TOOL_SUBST])



#Microsoft Visual Studio compiler selected?
if(env['CC']=='cl'):
   cflags.extend(['-EHsc'])
   debugcflags.extend(['-Zi'])
   debuglinkflags.extend(['/debug','/ASSEMBLYDEBUG'])
   if build_options['with_mkl']:
      env.Append(CPPDEFINES = ["EIGEN_USE_MKL_ALL"])
      cflags.extend([r"-IC:\Program Files (x86)\Intel\Composer XE\mkl\include"])
      linkflags.extend(["C:\Program Files (x86)\Intel\Composer XE\mkl\lib\intel64\mkl_intel_lp64.lib", "C:\Program Files (x86)\Intel\Composer XE\mkl\lib\intel64\mkl_core.lib", "C:\Program Files (x86)\Intel\Composer XE\mkl\lib\intel64\mkl_intel_thread.lib", "C:\Program Files (x86)\Intel\Composer XE\compiler\lib\intel64\libiomp5md.lib", "-ldl"])
else: 
   #slse? (clang / gcc settings are very similar)
   cflags.extend(['-fPIC'])
   #we require c++0x for smart pointers but presently not more than this.
   cxxflags.extend(['-std=c++0x'])
   releasecflags.extend(['-msse','-msse2'])         #extra compile flags for release
   releaselinkflags.extend(['-lstdc++'])
   debuglinkflags.extend(['-lstdc++'])
   debugcflags.extend(['-g','-Wextra'])
   if (build_options['CXXflags']):
    cxxflags.extend([build_options['CXXflags']])
    releaselinkflags.extend([build_options['CXXflags']])
    debuglinkflags.extend([build_options['CXXflags']])
   #cxxflags.extend(['-stdlib=libc++'])

#compile with zlib?
if build_options['with_zlib']:
  cflags.extend(['-DZLIB'])

env.Append(CCFLAGS=cflags)
env.Append(CXXFLAGS=cxxflags)
env.Append(LINKFLAGS=linkflags)
if build_options['CXX']:
   env['CXX'] = build_options['CXX']
if build_options['CC']:
   env['CC'] = build_options['CC']

if mymode == 'debug':
   env.Append(CCFLAGS=debugcflags)
   env.Append(LINKFLAGS=debuglinkflags)
else:
   env.Append(CCFLAGS=releasecflags)
   env.Append(LINKFLAGS=releaselinkflags)
#set shared library settings
#env.Append(SHLINK=ldshared)
#env.Append(SHLIBPREFIX="")
#set suffix for shared libraries
print env['SHLIBSUFFIX']
print env['SHLIBPREFIX']

env['SHLIBSUFFIX']=so_ext
env['SHLIBPREFIX'] = 'lib'

limix_include = ['#/src']
external_include = ['#/External']
env.Append(CPPPATH = limix_include)
env.Append(CPPPATH = external_include)

#record to extenral file?
if build_options['record']:
  env['SHCCCOM'] += " 2> %s" % build_options['record'] 
  env['SHCXXCOM'] += " 2> %s"% build_options['record'] 
  env['CCCOM'] += " 2> %s"% build_options['record'] 
  env['CXXCOM'] += " 2> %s"% build_options['record'] 


### 4. conf tests
conf = Configure(env)
#make sure the sconscripts can get to the variables
Export('env', 'conf','mymode','build_prefix','build_options','limix_include','external_include','limix_LIBS','limix_LIBS_str')

#put all .sconsign files in one place
env.SConsignFile()

#build external libraries
libnlopt=SConscript('External/nlopt/SConscript', variant_dir=os.path.join(build_prefix,'nlopt'),duplicate=0)
liblimix=SConscript('src/limix/SConscript',variant_dir=os.path.join(build_prefix,'limix'),duplicate=0)
limix_LIBS =  {'nlopt':libnlopt,'limix':liblimix}
limix_LIBS_str = ['limix','nlopt']

if build_options['with_zlib']:
   libzlib=SConscript('External/zlib/SConscript',variant_dir=os.path.join(build_prefix,'zlib'),duplicate=0)
   limix_LIBS['z'] = libzlib
   limix_LIBS_str.append('z')

#re-export to update libraries:
Export('limix_LIBS','limix_LIBS_str')

#build python interface?
if build_options['with_python']:
   python_interface=SConscript('src/interfaces/python/SConscript',variant_dir=os.path.join(build_prefix,'interfaces','python'),duplicate=0)

#build development cpp scripts
if build_options['with_developcpp']:
   command_line=SConscript('src/testing/SConscript',variant_dir=os.path.join(build_prefix,'testing'),duplicate=0)
   
#build documentation?
if build_options['with_documentation']:	
   print "option is obsolete. Update documentaiton manually using \"doc/update_doc.sh\""
   sys.exit(0)

#run unit tests?
if build_options['with_tests']:		
   args = [sys.executable, 'run_all.py',os.path.join('./..',build_prefix,'interfaces','python')]
   subprocess.call(args,cwd='tests')
   pass 

#install ?
if build_options['with_python']:
   #install limix module
   python_inst_lib = os.path.join(build_prefix,'interfaces','python','limix')
   python_inst_bin = os.path.join(build_prefix,'interfaces','python','bin')
   env.Depends(python_inst_lib, [python_interface])

   #install lib
   #env.Install(distutils.sysconfig.get_python_lib(),python_inst)
   #install bin
   inst_bin = os.path.join(distutils.sysconfig.PREFIX)
   inst_lib = distutils.sysconfig.get_python_lib()

   env.Install(inst_bin,python_inst_bin)
   ib = env.Alias('install-bin',inst_bin)
   il = env.Alias('install-lib',inst_lib)
   env.Alias('install',[ib,il])

   #python_inst = os.path.join(build_prefix,'interfaces','python','limix')
   #env.Depends(python_inst, [python_interface])
   #env.Alias('install',env.Install(distutils.sysconfig.get_python_lib(),python_inst))
   pass
   #install limix scripts
   #python_inst_bin = os.path.join(build_prefix,'interfaces','python','bin')
   #env.Depends(python_inst_bin, [python_interface])
   #env.Alias('install',env.Install(distutils.sysconfig.PREFIX,python_inst_bin))

