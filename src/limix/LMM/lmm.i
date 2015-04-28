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
//CLMMCore
%ignore CLMMCore;

//ALMM
%ignore ALMM::getPheno;
%ignore ALMM::getPv;
%ignore ALMM::getSnps;
%ignore ALMM::getCovs;
%ignore ALMM::getK;
%ignore ALMM::getPermutation;
%rename(getPheno) ALMM::agetPheno;
%rename(getPv) ALMM::agetPv;
%rename(getSnps) ALMM::agetSnps;
%rename(getCovs) ALMM::agetCovs;
%rename(getK) ALMM::agetK;
%rename(getPermutation) ALMM::agetPermutation;

//CLMM
//ignore C++ versions
%ignore CLMM::getNLL0;
%ignore CLMM::getNLLAlt;
%ignore CLMM::getLdeltaAlt;
%ignore CLMM::getLdelta0;
%ignore CLMM::getFtests;
%ignore CLMM::getLSigma;
%ignore CLMM::getBetaSNP;
%ignore CLMM::getBetaSNPste;
//rename argout versions for python; this overwrites the C++ convenience functions
%rename(getNLL0) CLMM::agetNLL0;
%rename(getNLLAlt) CLMM::agetNLLAlt;
%rename(getLdeltaAlt) CLMM::agetLdeltaAlt;
%rename(getLdelta0) CLMM::agetLdelta0;
%rename(getFtests) CLMM::agetFtests;
%rename(getLSigma) CLMM::agetLSigma;
%rename(getBetaSNP) CLMM::agetBetaSNP;
%rename(getBetaSNPste) CLMM::agetBetaSNPste;

//CInteractLMM
%ignore CInteractLMM::getInter;
%rename(getInter) CInteractLMM::agetInter;

}

//raw include
%include "limix/LMM/lmm.h"
