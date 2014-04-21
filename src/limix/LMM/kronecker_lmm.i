
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

}


//declare shared pointers
%shared_ptr(limix::CKroneckerLMM)
%shared_ptr(limix::CLMMKroneckerCore)

//raw include
%include "limix/LMM/kronecker_lmm.h"