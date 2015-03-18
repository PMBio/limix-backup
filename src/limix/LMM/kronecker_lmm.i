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

//KroneckerLMMCore
%ignore CLMMKroneckerCore;

//KroneckerLMM
%ignore CKroneckerLMM::getNLL0;
%ignore CKroneckerLMM::getNLLAlt;
%ignore CKroneckerLMM::getLdeltaAlt;
%ignore CKroneckerLMM::getLdelta0;
%rename(getNLL0) CKroneckerLMM::agetNLL0;
%rename(getNLLAlt) CKroneckerLMM::agetNLLAlt;
%rename(getLdeltaAlt) CKroneckerLMM::agetLdeltaAlt;
%rename(getLdelta0) CKroneckerLMM::agetLdelta0;
%rename(getBetaSNP) CKroneckerLMM::agetBetaSNP;
}


//declare shared pointers
%shared_ptr(limix::CKroneckerLMM)
%shared_ptr(limix::CLMMKroneckerCore)

//raw include
%include "limix/LMM/kronecker_lmm.h"
