# Find Python executable using which 

execute_process(
    COMMAND
    which python 
    OUTPUT_VARIABLE PYTHON_EXECUTABLE
    OUTPUT_STRIP_TRAILING_WHITESPACE
    )

mark_as_advanced (PYTHON_EXECUTABLE)