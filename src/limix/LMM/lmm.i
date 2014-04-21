// Copyright(c) 2014, The LIMIX developers (Christoph Lippert, Paolo Francesco Casale, Oliver Stegle)
// All rights reserved.
//
// LIMIX is provided under a 2-clause BSD license.
// See license.txt for the complete license.

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