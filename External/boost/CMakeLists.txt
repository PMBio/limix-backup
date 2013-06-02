#source files to be compiled for boost components used in limix
SET(SOURCES libs/iostreams/src/zlib.cpp
	    libs/iostreams/src/bzip2.cpp
	    libs/iostreams/src/gzip.cpp
	    libs/iostreams/src/mapped_file.cpp
	    libs/iostreams/src/file_descriptor.cpp
)

include_directories(./)
ADD_LIBRARY(boostlib STATIC ${SOURCES})


