#use systems path for building commands, etc.
import os
import sys

#import autoconfig like handling
from ACGenerateFile import *


#command line handling
#build mode
AddOption('--mode',dest='mode',
   type='string',nargs=1,action='store',default='release',help='Build mode: debug/release')

#build vcflib?
AddOption('--with-vcflib', dest='with_vcf', action='store_true',
help='Build VCFlib?', default=False)


#build options:
build_options= {}
build_options['with_vcf'] = GetOption('with_vcf')

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

cflags = ['']
debugcflags   = ['-g', '-Wextra', '-DDEBUG']   #extra compile flags for debug
releasecflags = ['-O2', '-DRELEASE']         #extra compile flags for release

if sys.platform=='win32':
   pass
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

if mymode == 'debug':
   env.Append(CCFLAGS=debugcflags)
else:
   env.Append(CCFLAGS=releasecflags)

limix_include = ['#/src']
external_include = ['#/External']

env.Append(CPPPATH = limix_include)
env.Append(CPPPATH = external_include)

#make sure the sconscripts can get to the variables
Export('env', 'mymode','build_prefix','build_options','limix_include','external_include')

#put all .sconsign files in one place
env.SConsignFile()

#build external libraries
nlopt=SConscript('External/nlopt/SConscript', variant_dir=os.path.join(build_prefix,'nlopt'),duplicate=0)

Export('nlopt')
limix=SConscript('src/limix/SConscript',variant_dir=os.path.join(build_prefix,'limix'),duplicate=0)

Export('limix','nlopt')
python_interface=SConscript('src/interfaces/python/SConscript',variant_dir=os.path.join(build_prefix,'interfaces','python'),duplicate=0)

