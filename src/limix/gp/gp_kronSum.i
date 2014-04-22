namespace limix {
//GPkrunSum

%ignore CGPkronSum::predictMean;
%ignore CGPkronSum::predictVar;
%rename(predictMean) CGPkronSum::apredictMean;
%rename(predictVar) CGPkronSum::apredictVar;
}

//raw include
%include "limix/gp/gp_kronSum.h"