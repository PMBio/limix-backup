// Copyright(c) 2014, The LIMIX developers(Christoph Lippert, Paolo Francesco Casale, Oliver Stegle)
//
//Licensed under the Apache License, Version 2.0 (the "License");
//you may not use this file except in compliance with the License.
//You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
//Unless required by applicable law or agreed to in writing, software
//distributed under the License is distributed on an "AS IS" BASIS,
//WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//See the License for the specific language governing permissions and
//limitations under the License.

namespace limix {
//CGPHyperParams
%ignore CGPHyperParams::get;
%ignore CGPHyperParams::getParamArray;
%rename(get) CGPHyperParams::aget;
%rename(getParamArray) CGPHyperParams::agetParamArray;
#ifdef SWIGPYTHON
%rename(__getitem__) CGPHyperParams::aget;
%rename(__setitem__) CGPHyperParams::set;
%rename(__str__) CGPHyperParams::toString;
#endif

//CGPbase
%ignore CGPbase::getX;
%ignore CGPbase::getY;
%ignore CGPbase::LMLgrad_covar;
%ignore CGPbase::LMLgrad_lik;
%ignore CGPbase::getParamArray;
%ignore CGPbase::predictMean;
%ignore CGPbase::predictVar;
%rename(getParamArray) CGPbase::agetParamArray;
%rename(getX) CGPbase::agetX;
%rename(getY) CGPbase::agetY;
%rename(LMLgrad_covar) CGPbase::aLMLgrad_covar;
%rename(LMLgrad_lik) CGPbase::aLMLgrad_lik;
%rename(LMLhess) CGPbase::aLMLhess;
%rename(LMLhess_covar) CGPbase::aLMLhess_covar;
%rename(LMLhess_lik) CGPbase::aLMLhess_lik;
%rename(LMLhess_covarlik) CGPbase::aLMLhess_covarlik;
%rename(getCov_laplace) CGPbase::agetCov_laplace;
%rename(getStd_laplace) CGPbase::agetStd_laplace;
%rename(predictMean) CGPbase::apredictMean;
%rename(predictVar) CGPbase::apredictVar;
}


//raw include
%include "limix/gp/gp_base.h"