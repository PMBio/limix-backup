namespace limix {

//CFreeFormCF
%ignore CFreeFormCF::getIparamDiag;
%ignore CFreeFormCF::K0Covar2Params;
%rename(getIparamDiag) CFreeFormCF::agetIparamDiag;
}

//raw include
%include "limix/covar/freeform.h"