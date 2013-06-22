#use systems path for building commands, etc.
import os

from ACGenerateFile import *
#from test import *

#get the mode flag from the command line
#default to 'release' if the user didn't specify
mymode = ARGUMENTS.get('mode', 'release')   #holds current mode

#check if the user has been naughty: only 'debug' or 'release' allowed
if not (mymode in ['debug', 'release']):
   print "Error: expected 'debug' or 'release', found: " + mymode
   Exit(1)

#tell the user what we're doing
print '**** Compiling in ' + mymode + ' mode...'

debugcflags = ['-W1', '-GX', '-fPIC', '-DDEBUG']   #extra compile flags for debug
releasecflags = ['-O2','-msse','-msse2','-fPIC', '-DRELEASE']         #extra compile flags for release

env = Environment(SHLIBPREFIX="",ENV = {'PATH' : os.environ['PATH']},tools = ['default', TOOL_SUBST],toolpath='.')
if mymode == 'debug':
   env.Append(CCFLAGS=debugcflags)
else:
   env.Append(CCFLAGS=releasecflags)

limix_include = ['#/src']
external_include = ['#/External']

env.Append(CPPPATH = limix_include)
env.Append(CPPPATH = external_include)

#make sure the sconscripts can get to the variables
Export('env', 'mymode', 'debugcflags', 'releasecflags','limix_include','external_include')

#put all .sconsign files in one place
env.SConsignFile()

#build external libraries
nlopt=SConscript('External/nlopt/SConscript', variant_dir=mymode+'/nlopt',duplicate=0)


#limix=SConscript('src/limix/SConscript',variant_dir=mymode+'/limix',duplicate=0)
#Export('limix')
#python_interface=SConscript('src/interfaces/python/SConscript',variant_dir=mymode+'/interfaces/python',duplicate=0)

