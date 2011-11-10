# Find Python libraries using distutils rather than other hacks
# PYTHON_LIBRARIES: dylib of python
# PYTHON_INCLUDE_DIRS: include directories
# PYTHONLIBS_FOUND, If false, do not try to use numpy headers.

if (NOT PYTHON_LIBRARIES OR NOT PYTHON_INCLUDE_DIRS)
    exec_program ("${PYTHON_EXECUTABLE}"
      ARGS "-c 'from distutils import sysconfig; print sysconfig.get_config_var(\"LDLIBRARY\")'"
      OUTPUT_VARIABLE PYTHON_LIBRARIES
      RETURN_VALUE PYTHONLIBS_FOUND)
    if (PYTHON_LIBRARIES MATCHES "Traceback")
    # Did not successfully include numpy
      set(PYTHONLIBS_FOUND FALSE)
    else ()
    # successful
      set (PYTHONLIBS_FOUND TRUE)
    endif ()

    exec_program ("${PYTHON_EXECUTABLE}"
      ARGS "-c 'from distutils import sysconfig; print sysconfig.get_python_inc()'"
      OUTPUT_VARIABLE PYTHON_INCLUDE_DIRS
      RETURN_VALUE PYTHONLIBS_FOUND)
    if (PYTHON_INCLUDE_DIRS MATCHES "Traceback")
    # Did not successfully include numpy
      set(PYTHONLIBS_FOUND FALSE)
    else ()
    # successful
      set (PYTHONLIBS_FOUND TRUE)
    endif ()


    mark_as_advanced (PYTHON_INCLUDE_DIRS)
    mark_as_advanced (PYTHON_LIBRARIES)
endif ()


