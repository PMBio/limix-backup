#use systems path for building commands, etc.
import os
import sys
import subprocess

#import autoconfig like handling
from ACGenerateFile import *


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

#build doxygen documentation?
AddOption('--with-documentation', dest='with_documentation', action='store_true',
help='Build doxygen documenation', default=False)

#run unit tests?
AddOption('--with-tests', dest='with_tests', action='store_true',
help='Run unit tests after build', default=False)

#build development tools for c++ (standalone snipsets)
AddOption('--with-developcpp', dest='with_developcpp', action='store_true',
help='Build development only commandline tools?', default=False)



#build options:
build_options= {}
build_options['with_vcf'] = GetOption('with_vcf')
build_options['with_python'] = GetOption('with_python')
build_options['with_developcpp'] = GetOption('with_developcpp')
build_options['with_tests'] = GetOption('with_tests')
build_options['with_documentation'] = GetOption('with_documentation')



### 2. build mode
#build mode:
mymode = GetOption('mode')
#check if the user has been naughty: only 'debug' or 'release' allowed
if not (mymode in ['debug', 'release']):
   print "Error: expected 'debug' or 'release', found: " + mymode
   Exit(1)

#tell the user what we're doing
#create build prefix
build_prefix = mymode + '.' + sys.platform
print '**** Compiling in ' + build_prefix + ' mode...'

cflags = []
linkflags = []
debugcflags   = ['-DDEBUG']   #extra compile flags for debug
releasecflags = ['-O2', '-DRELEASE']         #extra compile flags for release
debuglinkflags = []
releaselinkflags = []

### 3. compiler flags & environment
if sys.platform=='win32':
   cflags.extend(['-EHsc'])
   debugcflags.extend(['-Zi'])
   debuglinkflags.extend(['/debug','/ASSEMBLYDEBUG'])
else:
   cflags.extend(['-fPIC'])
   releasecflags.extend(['-msse','-msse2'])         #extra compile flags for release
   debugcflags.extend(['-g','-Wextra'])

#build environment
copy_env = ['PATH','INCLUDE','LIB','TMP']
ENV = {}
for key in copy_env:
    if key in os.environ.keys():
       ENV[key] = os.environ[key]

#TOOL_SUST
env = Environment(SHLIBPREFIX="",ENV=ENV,tools = ['default','doxygen',TOOL_SUBST])
env.Append(CCFLAGS=cflags)
env.Append(LINKFLAGS=linkflags)

if mymode == 'debug':
   env.Append(CCFLAGS=debugcflags)
   env.Append(LINKFLAGS=debuglinkflags)
else:
   env.Append(CCFLAGS=releasecflags)
   env.Append(LINKFLAGS=releaselinkflags)

limix_include = ['#/src']
external_include = ['#/External']
env.Append(CPPPATH = limix_include)
env.Append(CPPPATH = external_include)


### 4. conf tests
conf = Configure(env)
#hader checks
       
build_options['with_zlib'] = False

#if conf.CheckCHeader('zlib.h') & conf.CheckLib('libz'):
if False:
   build_options['with_zlib'] = True
else:
   build_options['with_zlib'] = False

if build_options['with_zlib']:
   env.Append(CCFLAGS=['-DZLIB'])

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
