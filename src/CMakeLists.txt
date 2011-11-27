#add external headers (mainly Eigen)


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
	    gpmix/covar/covariance.cpp
	    gpmix/covar/fixed.cpp	
	    gpmix/covar/fixed.cpp	
	    gpmix/covar/se.cpp
	    gpmix/covar/linear.cpp
	    gpmix/covar/dist.cpp
	    gpmix/utils/matrix_helper.cpp
	    gpmix/utils/beta.cpp
	    gpmix/utils/brentc.cpp
	    gpmix/utils/gamma.cpp
	    gpmix/utils/mathfunctions.cpp
	    gpmix/utils/fisherf.cpp
	    gpmix/LMM/lmm_old.cpp
	    gpmix/LMM/lmm.cpp	
	    gpmix/likelihood/likelihood.cpp
	    gpmix/gp/gp_base.cpp
)

ADD_LIBRARY(gpmixlib STATIC ${SOURCES})

INSTALL(TARGETS gpmixlib
   ARCHIVE DESTINATION lib
   LIBRARY DESTINATION lib)
