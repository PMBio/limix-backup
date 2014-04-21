namespace limix {
//ADataTerm

//ignore C++ versions
%ignore ADataTerm::evaluate();
%ignore ADataTerm::gradY();
%ignore ADataTerm::gradParamsRows();
%ignore ADataTerm::sumJacobianGradParams();
%ignore ADataTerm::sumLogJacobian();


//rename argout versions for python; this overwrites the C++ convenience functions
%rename(evaluate) ADataTerm::aEvaluate;
%rename(gradY) ADataTerm::aGradY;
%rename(gradParamsRows) ADataTerm::aGradParamsRows;
%rename(sumJacobianGradParams) ADataTerm::aSumJacobianGradParams;
%rename(sumLogJacobian) ADataTerm::aSumLogJacobian;
}

//raw include
%include "limix/mean/ADataTerm.h"