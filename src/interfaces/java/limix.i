%module core

%{
#define SWIG_FILE_WITH_INIT
#define SWIG
#include "limix/types.h"
#include "limix/lasso/lasso.h"
#include "limix/covar/covariance.h"
#include "limix/covar/linear.h"
#include "limix/covar/se.h"
#include "limix/covar/fixed.h"	
#include "limix/covar/freeform.h"	
#include "limix/covar/combinators.h"	
#include "limix/likelihood/likelihood.h"
#include "limix/mean/ADataTerm.h"
#include "limix/mean/CData.h"
#include "limix/mean/CLinearMean.h"
#include "limix/mean/CSumLinear.h"
#include "limix/mean/CKroneckerMean.h"
#include "limix/gp/gp_base.h"
#include "limix/gp/gp_kronecker.h"
#include "limix/gp/gp_opt.h"
#include "limix/LMM/lmm.h"
#include "limix/LMM/kronecker_lmm.h"
#include "limix/LMM/CGPLMM.h"
#include "limix/modules/CVqtl.h"
#include "limix/modules/CMultiTraitVQTL.h"


using namespace limix;
//  removed namespace bindings (12.02.12)
%}

/* Get the numpy typemaps */
//%include "numpy.i"
//support for eigen matrix stuff
//%include "eigen.i"
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


// Includ dedicated interface files
%include "./../covar.i"
%include "./../gp.i"
%include "./../lik.i"
%include "./../mean.i"
%include "./../lmm.i"
%include "./../modules.i"


//generated outodoc:
//%feature("autodoc", "1")
%include "limix/types.h"
%include "limix/lasso/lasso.h"
%include "limix/covar/covariance.h"
%include "limix/covar/linear.h"
%include "limix/covar/se.h"
%include "limix/covar/fixed.h"
%include "limix/covar/freeform.h"	
%include "limix/covar/combinators.h"	
%include "limix/likelihood/likelihood.h"
%include "limix/mean/ADataTerm.h"
%include "limix/mean/CData.h"
%include "limix/mean/CLinearMean.h"
%include "limix/mean/CSumLinear.h"
%include "limix/mean/CKroneckerMean.h"
%include "limix/gp/gp_base.h"
%include "limix/gp/gp_kronecker.h"
%include "limix/gp/gp_opt.h"
%include "limix/LMM/lmm.h"
%include "limix/LMM/CGPLMM.h"
%include "limix/LMM/kronecker_lmm.h"
%include "limix/modules/CVqtl.h"
%include "limix/modules/CMultiTraitVQTL.h"



 
