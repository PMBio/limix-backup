#add external headers (mainly Eigen)
include_directories(../External)
include_directories(./)


if(DEBUG_GPMIX)
  ADD_DEFINITIONS(-DOSSOLOG_COMPILE -DOSSOLOG_STDERR -DOSSOLOG_DEBUG)
else(DEBUG_GPMIX)
  if(SILENT_GPMIX)
    #do nothing: don't include OSSOLOG_COMPILE
  else(SILENT_GPMIX)
    ADD_DEFINITIONS(-DOSSOLOG_COMPILE -DOSSOLOG_STDERR) 
  endif(SILENT_GPMIX)
endif(DEBUG_GPMIX)

SET(SOURCES gpmix/covar/covariance.cpp
	    gpmix/covar/se.cpp
	    gpmix/covar/linear.cpp
	    gpmix/covar/dist.cpp
	    gpmix/matrix/matrix_helper.cpp
	    gpmix/gp/gp_base.cpp
)

ADD_LIBRARY(gpmixlib STATIC ${SOURCES})

INSTALL(TARGETS gpmixlib
   ARCHIVE DESTINATION lib
   LIBRARY DESTINATION lib)
