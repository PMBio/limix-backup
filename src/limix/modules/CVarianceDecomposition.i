namespace limix {

//AVarianceTerm
%rename(getK) AVarianceTerm::agetK;
%rename(getX) AVarianceTerm::agetX;
%rename(getScales) AVarianceTerm::agetScales;

//CvarianceDecomposition
%rename(getScales) CVarianceDecomposition::agetScales;

}

//raw include
%include "limix/modules/CVarianceDecomposition.h"