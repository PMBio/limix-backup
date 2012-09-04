cmake_minimum_required(VERSION 2.8)
project(LIMIX)
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

#build options
#PYTHON off by default
option(BUILD_PYTHON_PACKAGE "Enable Python bindings" OFF)
#R optional
option(BUILD_R_PACKAGE "Enable R bindings" OFF)
#profiling using google perftools (currently only python .so)
option(GOOGLE_PERF "Enable google perftools support" OFF)
#architecture secific optimization
option(OPTIMIZE_ARCHITECTURE "Enanle architecture specific optimizations" OFF)

#build stuff
option(BUILD_UNIVERSAL "Build both 32 and 64 bit versions" OFF)

#compile i386+x86_64 universal binaries on OS X
if (BUILD_UNIVERSAL)
IF(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
SET(CMAKE_OSX_ARCHITECTURES "i386;x86_64")
ENDIF(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
endif (BUILD_UNIVERSAL)

#set architecture manually (cluster)
set(TARGET_ARCHITECTURE "auto")


#optimization for architecture:
include (OptimizeForArchitecture)
include (AddCompilerFlag)

## Documentation for final package
# manual installation paths

##PYTHON build options##
# Path to python executable. If specified peer is built against this
# python version, otherwise "which python" is used for auto-detection
#SET(PYTHON_EXECUTABLE "/c/python27/python.exe")
#Python include directory 
#normally there is no need to specify them;instead these are
#auto-determined from the location of the binary.
#SET(PYTHON_INCLUDE_DIRS "")
#Python install directory
#If specified, the peer module will be installed here. Otherwise the
#default site-packages location of the respective python installation
#is used. 
SET(PYTHON_INSTDIR "/c/Python27")
#Numpy header fiels.
#Usually there is no need to specify them as they are automatically
#determined from the python binary.
#Note: it is vital that peer is built against the exact same python
#installation that you are aiming to use.
#SET(NUMPY_INCLUDE_DIR "")
#
##/PYTHON END##


##R build options###
#set(R_EXECUTABLE "")
#R executable 
#set(R_HOME "")
#R home directory
#set(R_H "")
#R include directory for headers
#set(R_INCLUDEDIR "")
#R include dir
#set(LIBR_SO)
#R dylib or dynamic link SO 
##/R END##


# OLI's build options
set(LIBR_SO "/opt/local/lib/R/lib/x86_64/libR.dylib")
##/R END

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

#There are lots of scripts with cmake
#for finding external libraries. 
#see /usr/local/share/cmake-2.6/Modules/Find*.cmake for more examples

find_package(SWIG REQUIRED)
include(${SWIG_USE_FILE})

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_SWIG_FLAGS "")

set(CMAKE_MODULE_PATH "${LIMIX_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

#add_definitions(-g -Wextra -D_BSD_SOURCE)

#optimize automatically for architecture:

if (OPTIMIZE_ARCHITECTURE)
OptimizeForArchitecture()
else (OPTIMIZE_ARCHITECTURE)
AddCompilerFlag("-march=i686")
AddCompilerFlag("-msse")
AddCompilerFlag("-msse2")
endif (OPTIMIZE_ARCHITECTURE)

AddCompilerFlag("-fPIC")
AddCompilerFlag("-O2")



#includes
include_directories(./External)
include_directories(./src)

#add Cmake modules
#nlopt
add_subdirectory(./External/nlopt)
#LIMIX source
add_subdirectory(./src)

#### build external interface (optional) #####
if (BUILD_PYTHON_PACKAGE)
add_subdirectory(./src/interfaces/python)
endif (BUILD_PYTHON_PACKAGE)
if (BUILD_R_PACKAGE)
add_subdirectory(./src/interfaces/R)
endif (BUILD_R_PACKAGE)

