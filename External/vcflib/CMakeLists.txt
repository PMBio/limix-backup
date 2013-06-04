
#source files to be compiled for VCFlib
SET(SOURCES Variant.cpp
                      split.cpp
		      fastahack/Fasta.cpp		   
		      smithwaterman/SmithWatermanGotoh.cpp
		      smithwaterman/Repeats.cpp
		      smithwaterman/IndelAllele.cpp
		      smithwaterman/disorder.cpp
		      smithwaterman/LeftAlign.cpp
		      fsom/fsom.c
		      tabixpp/tabix.cpp
		      tabixpp/bgzf.c
		      tabixpp/index.c
		      ssw.c
		      ssw_cpp.cpp		      		     
)

include_directories(./)
ADD_LIBRARY(vcflib STATIC ${SOURCES})


