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
reswig = True
reswig = False 


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
AddOption('--with-vcflib', dest='with_vcf', action='store_true',
help='Build VCFlib?', default=True)

#build python interface?
AddOption('--without-python', dest='with_python', action='store_false',
help='Disable python interface', default=True)

#run swig?
AddOption('--reswig', dest='reswig', action='store_true', help='Run swig?', default=reswig)

#build doxygen documentation?
AddOption('--with-documentation', dest='with_documentation', action='store_true',
help='Build doxygen documenation', default=False)

#run unit tests?
AddOption('--with-tests', dest='with_tests', action='store_true',
help='Run unit tests after build', default=False)

#build development tools for c++ (standalone snipsets)
AddOption('--with-developcpp', dest='with_developcpp', action='store_true',
help='Build development only commandline tools?', default=False)

#override  build settings
AddOption('--CXX',dest='CXX',type='string',nargs=1,action='store',default=cxx,help='Manual specified CXX')
#override CC compiler (for open MPI)
AddOption('--CC',dest='CC',type='string',nargs=1,action='store',default=cc,help='Manual specified CC')
#override  build tool
AddOption('--build_tool',dest='build_tool',type='string',nargs=1,action='store',default=build_tool,help='Manual specification of build tool')
#use static bind?
#note: if enabled, the library is effectively GPL as nlopt is lesser GPL and hence we require dynamic binding
AddOption('--static-bind',dest='static_bind',action='store_true',help='Enforce static binding of external libs',default=False)

#### options that are needed for distutils compatibility ####
#set output?
AddOption('--record',dest='record',type='string',nargs=1,action='store',default='',help='Record build output for distutils build')

#dummpy options to please setup tools
AddOption('--compile', dest='compile', action='store_true', help='No action', default=reswig)
AddOption('--single-version-externally-managed', dest='single-version-externally-managed', action='store_true', help='No action', default=reswig)

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
build_options['with_vcf'] = GetOption('with_vcf')
build_options['with_python'] = GetOption('with_python')
build_options['with_developcpp'] = GetOption('with_developcpp')
build_options['with_tests'] = GetOption('with_tests')
build_options['with_documentation'] = GetOption('with_documentation')
build_options['CXX'] = GetOption('CXX')
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

#build environment
copy_env = ['PATH','INCLUDE','LIB','TMP']
ENV = {}
for key in copy_env:
    if key in os.environ.keys():
       ENV[key] = os.environ[key]

#TOOL_SUST
env = Environment(SHLIBPREFIX="",ENV=ENV,tools = [build_options['build_tool'],'doxygen',TOOL_SUBST])
#env = Environment(SHLIBPREFIX="",ENV=ENV,tools = [build_options['build_tool'],TOOL_SUBST])

#Microsoft Visual Studio compiler selected?
if(env['CC']=='cl'):
   cflags.extend(['-EHsc'])
   debugcflags.extend(['-Zi'])
   debuglinkflags.extend(['/debug','/ASSEMBLYDEBUG'])
else: 
   #slse? (clang / gcc settings are very similar)
   cflags.extend(['-fPIC'])
   #we require c++0x for smart pointers but presently not more than this.
   cxxflags.extend(['-std=c++0x'])
   #cxxflags.extend(['-stdlib=libc++'])
   # releasecflags.extend(['-msse','-msse2','-fopenmp'])         #extra compile flags for release
   releasecflags.extend(['-msse','-msse2'])         #extra compile flags for release
   # releaselinkflags.extend(['-lgomp'])
   releaselinkflags.extend(['-lstdc++'])
   debuglinkflags.extend(['-lstdc++'])
   debugcflags.extend(['-g','-Wextra'])

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
#hader checks

#make sure the sconscripts can get to the variables
Export('env', 'conf','mymode','build_prefix','build_options','limix_include','external_include')

#put all .sconsign files in one place
env.SConsignFile()

#build external libraries
libnlopt=SConscript('External/nlopt/SConscript', variant_dir=os.path.join(build_prefix,'nlopt'),duplicate=0)

Export('libnlopt')
liblimix=SConscript('src/limix/SConscript',variant_dir=os.path.join(build_prefix,'limix'),duplicate=0)

#build python interface?
if build_options['with_python']:
   Export('liblimix','libnlopt')
   python_interface=SConscript('src/interfaces/python/SConscript',variant_dir=os.path.join(build_prefix,'interfaces','python'),duplicate=0)

#build development cpp scripts
if build_options['with_developcpp']:
   Export('liblimix','libnlopt')
   command_line=SConscript('src/testing/SConscript',variant_dir=os.path.join(build_prefix,'testing'),duplicate=0)
   
#build documentation?
if build_options['with_documentation']:	
   doxy=env.Doxygen("./doc/doxy.cfg")
   AlwaysBuild([doxy])
   pass

#run unit tests?
if build_options['with_tests']:		
   args = [sys.executable, 'run_all.py',os.path.join('./..',build_prefix,'interfaces','python')]
   subprocess.call(args,cwd='tests')
   pass 

#install ?
if build_options['with_python']:
   python_inst = os.path.join(build_prefix,'interfaces','python','limix')
   env.Depends(python_inst, [python_interface])
   env.Alias('install',env.Install(distutils.sysconfig.get_python_lib(),python_inst))
