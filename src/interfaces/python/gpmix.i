%module gpmix

%{
#define SWIG_FILE_WITH_INIT
#define SWIG
#include "gpmix/types.h"
#include "gpmix/LMM/lmm.h"
#include "gpmix/LMM/kronecker_lmm.h"
#include "gpmix/lasso/lasso.h"
#include "gpmix/covar/covariance.h"
#include "gpmix/covar/linear.h"
#include "gpmix/covar/se.h"
#include "gpmix/covar/fixed.h"	
#include "gpmix/covar/freeform.h"	
#include "gpmix/covar/combinators.h"	
#include "gpmix/likelihood/likelihood.h"
#include "gpmix/mean/ADataTerm.h"
#include "gpmix/mean/CData.h"
#include "gpmix/mean/CLinearMean.h"
#include "gpmix/mean/CKroneckerMean.h"
#include "gpmix/gp/gp_base.h"
#include "gpmix/gp/gp_kronecker.h"
#include "gpmix/gp/gp_opt.h"

using namespace gpmix;
//  removed namespace bindings (12.02.12)
%}

/* Get the numpy typemaps */
%include "numpy.i"
//support for eigen matrix stuff
%include "eigen.i"
//support for std libs
//suport for std_shared pointers in tr1 namespace

#define SWIG_SHARED_PTR_NAMESPACE std
#define SWIG_SHARED_PTR_SUBNAMESPACE tr1
%include "std_shared_ptr.i"
%include "std_vector.i"
%include "std_map.i"
%include "std_string.i"


%init %{
  import_array();
%}


//%shared_ptr(gpmix::CTest)
%include "covar.i"
%include "gp.i"
%include "lik.i"
%include "mean.i"
%include "lmm.i"


//generated outodoc:
//%feature("autodoc", "1")
%include "gpmix/types.h"
%include "gpmix/LMM/lmm.h"
%include "gpmix/LMM/kronecker_lmm.h"
%include "gpmix/lasso/lasso.h"
%include "gpmix/covar/covariance.h"
%include "gpmix/covar/linear.h"
%include "gpmix/covar/se.h"
%include "gpmix/covar/fixed.h"
%include "gpmix/covar/freeform.h"	
%include "gpmix/covar/combinators.h"	
%include "gpmix/likelihood/likelihood.h"
%include "gpmix/mean/ADataTerm.h"
%include "gpmix/mean/CData.h"
%include "gpmix/mean/CLinearMean.h"
%include "gpmix/mean/CKroneckerMean.h"
%include "gpmix/gp/gp_base.h"
%include "gpmix/gp/gp_kronecker.h"
%include "gpmix/gp/gp_opt.h"



 
