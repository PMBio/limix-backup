# Find R executable using which 

execute_process(
    COMMAND
    which R 
    OUTPUT_VARIABLE R_EXECUTABLE
    OUTPUT_STRIP_TRAILING_WHITESPACE
    )

mark_as_advanced (R_EXECUTABLE)
