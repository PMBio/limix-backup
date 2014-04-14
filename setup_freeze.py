"""LIMIX setup.py for cx-freeze functionality and other bits"""

import sys
from cx_Freeze import setup, Executable

exclude_modules = ["tkinter","tcl","md5"]


#general includes
#include_modules = ["md5","IPython","numpy","scipy",
include_modules = ["numpy","scipy",
	"pylab","scipy.stats","scipy.special","scipy.special._ufuncs",
	"scipy.special._ufuncs_cxx","scipy.sparse",
	"scipy.sparse.linalg","scipy.linalg","scipy.sparse.csgraph",
	"scipy.sparse.csgraph._validation","scipy.sparse.linalg.dsolve.umfpack",
	"scipy.integrate._ode","scipy.integrate.vode","scipy.integrate.lsoda"]
#limix related includes and modules
include_modules.extend(["limix","limix.modules.qtl"])

#include files(just for testing)
include_files = ["limix_script.py"]

# Dependencies are automatically detected, but it might need fine tuning.
# build_exe_options = {"packages": ["os"], "includes": ["h5py._errors"],"excludes": ["tkinter"]}
build_exe_options = {"packages": ["os"], "includes": include_modules,"excludes": exclude_modules,'include_files':include_files}

# GUI applications require a different base on Windows (the default is for a
# console application).
base = None

setup(  name = "guifoo",
        version = "0.1",
        description = "My GUI application!",
        options = {"build_exe": build_exe_options},
        executables = [Executable("limix_main.py", 
        	base=base,
        	compress=True,
        	copyDependentFiles = True,
        	appendScriptToExe = False,
        	appendScriptToLibrary = False,
        	icon = None
        	)])
