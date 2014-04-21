
namespace limix {
//CGPSum
%ignore CGPSum::predictMean;
%ignore CGPSum::predictVar;
%rename(predictMean) CGPSum::apredictMean;
%rename(predictVar) CGPSum::apredictVar;

}

//raw include
%include "limix/gp/gp_Sum.h"