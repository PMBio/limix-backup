#use systems path for building commands, etc.
import os
import sys

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

#build command line tools (currently only for debugging)
AddOption('--with-commandline', dest='with_commandline', action='store_true',
help='Build Command line tools?', default=False)



#build options:
build_options= {}
build_options['with_vcf'] = GetOption('with_vcf')
build_options['with_python'] = GetOption('with_python')
build_options['with_commandline'] = GetOption('with_commandline')


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
debugcflags   = ['-g', '-Wextra', '-DDEBUG']   #extra compile flags for debug
releasecflags = ['-O2', '-DRELEASE']         #extra compile flags for release

### 3. compiler flags & environment
if sys.platform=='win32':
   cflags.extend(['-EHsc'])
else:
   cflags.extend(['-fPIC'])
   releasecflags.extend(['-msse','-msse2'])         #extra compile flags for release

#build environment
copy_env = ['PATH','INCLUDE','LIB','TMP']
ENV = {}
for key in copy_env:
    if key in os.environ.keys():
       ENV[key] = os.environ[key]

env = Environment(SHLIBPREFIX="",ENV=ENV,tools = ['default', TOOL_SUBST],toolpath='.')
env.Append(CCFLAGS=cflags)
env.Append(LINKFLAGS=linkflags)

if mymode == 'debug':
   env.Append(CCFLAGS=debugcflags)
else:
   env.Append(CCFLAGS=releasecflags)

limix_include = ['#/src']
external_include = ['#/External']
env.Append(CPPPATH = limix_include)
env.Append(CPPPATH = external_include)


### 4. conf tests
conf = Configure(env)
#hader checks

if conf.CheckCHeader('zlib.h') & conf.CheckLib('libz'):
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

#build command line?
if build_options['with_commandline']:
   Export('liblimix','libnlopt')
   command_line=SConscript('src/testing/SConscript',variant_dir=os.path.join(build_prefix,'testing'),duplicate=0)
   
