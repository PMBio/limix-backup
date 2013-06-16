#PYTHON build system using SWIG

find_package(SWIG REQUIRED)

#default: enable python bindings

#1. find swig
INCLUDE(${SWIG_USE_FILE})

#find python
if(NOT PYTHON_EXECUTABLE)
       FIND_PACKAGE(PythonInterpWhich)
endif()

#we got everything?
if ( (NOT EXISTS ${PYTHON_EXECUTABLE} ) )
   message("Did not find python")
   set(BUILD_PYTHON_PACKAGE OFF CACHE BOOL "Enable Python bindings" FORCE)
endif()

#find python libs and includes
if (NOT PYTHON_LIBRARIES OR NOT PYTHON_INCLUDE_DIRS)
   FIND_PACKAGE(PythonLibsDist)
endif()

#find dist-packages
if (NOT PYTHON_INSTDIR)
   FIND_PACKAGE(PythonInst)
endif()

#find numpy
if (NOT NUMPY_INCLUDE_DIR)
   FIND_PACKAGE(Numpy)
endif()

if ( (NOT EXISTS ${PYTHON_EXECUTABLE}))
   message("Did not find python executable, disabling python interface.")
   message("Try specifying executable manually in CMakeLists.txt")
   set(BUILD_PYTHON_PACKAGE OFF CACHE BOOL "Enable Python bindings" FORCE)
endif()

if ( (NOT EXISTS ${PYTHON_INCLUDE_DIRS}))
   message("Did not find python include dirs")
   message(${PYTHON_INCLUDE_DIRS})
   set(BUILD_PYTHON_PACKAGE OFF CACHE BOOL "Enable Python bindings" FORCE)
endif()


if ( (NOT EXISTS ${NUMPY_INCLUDE_DIR}) )
   message("Did not find numpy, disabling python interface")
   message(${NUMPY_INCLUDE_DIR})
   set(BUILD_PYTHON_PACKAGE OFF CACHE BOOL "Enable Python bindings" FORCE)
endif()


if ( NOT EXISTS ${PYTHON_INSTDIR})
   message("Did not find python install directory, disabling python interface")
   set(BUILD_PYTHON_PACKAGE OFF CACHE BOOL "Enable Python bindings" FORCE)
endif()



if(BUILD_PYTHON_PACKAGE)



include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${NUMPY_INCLUDE_DIR})


include_directories(./../)


set(CMAKE_SWIG_FLAGS "")

#add google profiling code if enable:
if (GOOGLE_PERF)
SET(SWIGLINKLIBS profiler)
endif (GOOGLE_PERF)

set_source_files_properties(limix.i PROPERTIES CPLUSPLUS ON)
set_source_files_properties(limix.i PROPERTIES SWIG_FLAGS "-includeall")
set_source_files_properties(limix.i PROPERTIES SWIG_FLAGS "-cpperraswarn")
swig_add_module(core python limix.i)
swig_link_libraries(core limixlib nlopt ${SWIGLINKLIBS} ${ZLIB_LIBRARIES})

IF(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")	 
  # Mac OS X specific code
  # Without this flag it does not find some of the libraries
  SET(CMAKE_MODULE_LINKER_FLAGS "-undefined dynamic_lookup")
ENDIF(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")

ADD_CUSTOM_COMMAND(
    TARGET _core POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
       ${LIMIX_SOURCE_DIR}/src/interfaces/python/limix
       ${CMAKE_CURRENT_BINARY_DIR}/limix)

ADD_CUSTOM_COMMAND(
    TARGET _core POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_CURRENT_BINARY_DIR}/core.py
        ${CMAKE_CURRENT_BINARY_DIR}/limix/core.py)

ADD_CUSTOM_COMMAND(
    TARGET _core POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_CURRENT_BINARY_DIR}/_core.so
        ${CMAKE_CURRENT_BINARY_DIR}/limix/_core.so)

#Install python .so and python wrapper
INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/_limix.so DESTINATION ${PYTHON_INSTDIR})
INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/limix.py DESTINATION ${PYTHON_INSTDIR})
#install demo into the build directory

endif(BUILD_PYTHON_PACKAGE)
