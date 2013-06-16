#use systems path for building commands, etc.
import os

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

env = Environment(SHLIBPREFIX="",ENV = {'PATH' : os.environ['PATH']},)
if mymode == 'debug':
   env.Append(CCFLAGS=debugcflags)
else:
   env.Append(CCFLAGS=releasecflags)

env.Append(CPPPATH = ['#/src','#/External'])

#make sure the sconscripts can get to the variables
Export('env', 'mymode', 'debugcflags', 'releasecflags')

#put all .sconsign files in one place
env.SConsignFile()

#build external libraries
#project = 'nlopt'
#nlopt=SConscript('External/nlopt/SConscript', exports=['project'])

project = 'limix'
limix=SConscript('src/SConscript', exports=['project'],variant_dir=mymode,duplicate=0)

#project = 'python_module'
#SConscript('src/interfaces/python/SConscript', exports=['project'])

