// Copyright(c) 2014, The LIMIX developers(Christoph Lippert, Paolo Francesco Casale, Oliver Stegle)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
%module core
%feature("autodoc", "3");
%include exception.i       

%{
#define SWIG_FILE_WITH_INIT
#define SWIG
#include "limix/types.h"
#include "limix/covar/covariance.h"
#include "limix/utils/cache.h"
#include "limix/covar/linear.h"
#include "limix/covar/freeform.h"
#include "limix/covar/se.h"
#include "limix/covar/combinators.h"	
#include "limix/likelihood/likelihood.h"
#include "limix/mean/ADataTerm.h"
#include "limix/mean/CData.h"
#include "limix/mean/CLinearMean.h"
#include "limix/mean/CSumLinear.h"
#include "limix/mean/CKroneckerMean.h"
#include "limix/gp/gp_base.h"
#include "limix/gp/gp_kronecker.h"
#include "limix/gp/gp_kronSum.h"
#include "limix/gp/gp_Sum.h"
#include "limix/gp/gp_opt.h"
#include "limix/LMM/lmm.h"
#include "limix/LMM/kronecker_lmm.h"
#include "limix/modules/CVarianceDecomposition.h"
#include "limix/io/dataframe.h"
#include "limix/io/genotype.h"
#include "limix/LMM_forest/lmm_forest.h"

using namespace limix;
//  removed namespace bindings (12.02.12)
%}

/* Get the numpy typemaps */
%include "numpy.i"
//support for eigen matrix stuff
%include "eigen.i"
//include typemaps
%include "typemaps.i"

#define SWIG_SHARED_PTR_NAMESPACE std
//C11, no tr!
//#define SWIG_SHARED_PTR_SUBNAMESPACE tr1
%include "std_shared_ptr.i"

//removed boost
//%include <boost_shared_ptr.i>

%include "std_vector.i"
%include "std_map.i"
%include "std_string.i"
%include "stdint.i"


%init %{
  import_array();
%}


%exception{
	try {
	$action
	} catch (limix::CLimixException& e) {
	std::string s("LIMIX error: "), s2(e.what());
	s = s + s2;
	SWIG_exception(SWIG_RuntimeError, s.c_str());
	return NULL;
	} catch (...) {
	SWIG_exception(SWIG_RuntimeError,"Unknown exception");
	}
}


// Includ dedicated interface files
/* Note: currently these only contain definitions of shared pointers. We should move these into interface files below as soon as possible
*/
%include "./../types.i"
%include "./../covar.i"
%include "./../lik.i"
%include "./../mean.i"
%include "./../lmm.i"
%include "./../gp.i"
%include "./../modules.i"
%include "./../io.i"

//interface files:
%include "limix/types.i"
%include "limix/utils/cache.i"
%include "limix/covar/covariance.i"
%include "limix/covar/linear.i"
%include "limix/covar/freeform.i"
%include "limix/covar/se.i"
%include "limix/covar/combinators.i"	
%include "limix/likelihood/likelihood.i"
%include "limix/mean/ADataTerm.i"
%include "limix/mean/CData.i"
%include "limix/mean/CLinearMean.i"
%include "limix/mean/CSumLinear.i"
%include "limix/mean/CKroneckerMean.i"
%include "limix/gp/gp_base.i"
%include "limix/gp/gp_kronecker.i"
%include "limix/gp/gp_kronSum.i"
%include "limix/gp/gp_Sum.i"
%include "limix/gp/gp_opt.i"
%include "limix/LMM/lmm.i"
%include "limix/LMM/kronecker_lmm.i"
%include "limix/modules/CVarianceDecomposition.i"
%include "limix/io/dataframe.i"
%include "limix/io/genotype.i"
%include "limix/LMM_forest/lmm_forest.i"
