#JAVA build system using SWIG

find_package(SWIG REQUIRED)

#1. find swig
INCLUDE(${SWIG_USE_FILE})

if(BUILD_JAVA_PACKAGE)


#find python
if(NOT BUILD_PYTHON_PACKAGE)
       FIND_PACKAGE(JNI)
endif()

include_directories(${JAVA_INCLUDE_PATH})


include_directories(./../)

set(CMAKE_SWIG_FLAGS "")

#add google profiling code if enable:
if (GOOGLE_PERF)
SET(SWIGLINKLIBS profiler)
endif (GOOGLE_PERF)

set_source_files_properties(limix.i PROPERTIES CPLUSPLUS ON)
set_source_files_properties(limix.i PROPERTIES SWIG_FLAGS "-includeall")
set_source_files_properties(limix.i PROPERTIES SWIG_FLAGS "-cpperraswarn")
swig_add_module(core java limix.i)
swig_link_libraries(core limixlib nlopt ${SWIGLINKLIBS})

IF(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")	 
  # Mac OS X specific code
  # Without this flag it does not find some of the libraries
  SET(CMAKE_MODULE_LINKER_FLAGS "-undefined dynamic_lookup")
ENDIF(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")

#ADD_CUSTOM_COMMAND(
#   TARGET _core POST_BUILD
#    COMMAND ${CMAKE_COMMAND} -E copy_directory
#       ${LIMIX_SOURCE_DIR}/src/interfaces/python/limix
#       ${CMAKE_CURRENT_BINARY_DIR}/limix)

#ADD_CUSTOM_COMMAND(
#    TARGET _core POST_BUILD
#    COMMAND ${CMAKE_COMMAND} -E copy
#        ${CMAKE_CURRENT_BINARY_DIR}/core.py
#        ${CMAKE_CURRENT_BINARY_DIR}/limix/core.py)

#ADD_CUSTOM_COMMAND(
#    TARGET _core POST_BUILD
#    COMMAND ${CMAKE_COMMAND} -E copy
#        ${CMAKE_CURRENT_BINARY_DIR}/_core.so
#        ${CMAKE_CURRENT_BINARY_DIR}/limix/_core.so)

#Install python .so and python wrapper
#INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/_limix.so DESTINATION ${PYTHON_INSTDIR})
#INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/limix.py DESTINATION ${PYTHON_INSTDIR})
#install demo into the build directory

endif(BUILD_JAVA_PACKAGE)
