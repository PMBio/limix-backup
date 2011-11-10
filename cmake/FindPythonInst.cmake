# Find Python install Directory


execute_process(
    COMMAND
    ${PYTHON_EXECUTABLE} -c "from distutils import sysconfig; print sysconfig.get_python_lib()"
    OUTPUT_VARIABLE PYTHON_INSTDIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    )

mark_as_advanced (PYTHON_INSTDIR)