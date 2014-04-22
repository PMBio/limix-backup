
namespace limix {
//CGPkronecker

%ignore CGPkronecker::predictMean;
%ignore CGPkronecker::predictVar;
%rename(predictMean) CGPkronecker::apredictMean;
%rename(predictVar) CGPkronecker::apredictVar;

}

//raw include
%include "limix/gp/gp_kronecker.h"