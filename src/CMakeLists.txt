#add external headers (mainly Eigen)


if(DEBUG_LIMIX)
  ADD_DEFINITIONS(-DOSSOLOG_COMPILE -DOSSOLOG_STDERR -DOSSOLOG_DEBUG)
else(DEBUG_LIMIX)
  if(SILENT_LIMIX)
    #do nothing: don't include OSSOLOG_COMPILE
  else(SILENT_LIMIX)
    ADD_DEFINITIONS(-DOSSOLOG_COMPILE -DOSSOLOG_STDERR) 
  endif(SILENT_LIMIX)
endif(DEBUG_LIMIX)

SET(SOURCES limix/covar/covariance.cpp
	    limix/covar/covariance.cpp
	    limix/covar/freeform.cpp	
	    limix/covar/se.cpp
	    limix/covar/combinators.cpp
	    limix/covar/linear.cpp
	    limix/covar/dist.cpp
	    limix/utils/matrix_helper.cpp
	    limix/utils/beta.cpp
	    limix/utils/brentc.cpp
	    limix/utils/gamma.cpp
	    limix/utils/mathfunctions.cpp
	    limix/utils/fisherf.cpp
	    limix/LMM/lmm_old.cpp
	    limix/LMM/lmm.cpp	
	    limix/LMM/kronecker_lmm.cpp	
	    limix/LMM/CGPLMM.cpp	
	    limix/lasso/lasso.cpp
	    limix/likelihood/likelihood.cpp
	    limix/gp/gp_base.cpp
	    limix/gp/gp_kronecker.cpp
	    limix/gp/gp_kronSum.cpp
	    limix/gp/gp_Sum.cpp
	    limix/gp/gp_opt.cpp
	    limix/mean/ADataTerm.cpp
	    limix/mean/CLinearMean.cpp
	    limix/mean/CSumLinear.cpp
	    limix/mean/CKroneckerMean.cpp
	    limix/mean/CData.cpp
	    limix/modules/CVqtl.cpp
	    limix/modules/CMultiTraitVQTL.cpp
	    limix/modules/CVarianceDecomposition.cpp
	    limix/io/genotype.cpp
	    limix/io/dataframe.cpp
	    limix/io/split.cpp
)


ADD_LIBRARY(limixlib STATIC ${SOURCES})

INSTALL(TARGETS limixlib
   ARCHIVE DESTINATION lib
   LIBRARY DESTINATION lib)
