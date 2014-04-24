namespace limix {
//genotype

%ignore CGenotypeBlock::getPosition;
%ignore CGenotypeBlock::getMatrix;
%rename(getMatrix) CGenotypeBlock::agetMatrix;
%rename(getPosition) CGenotypeBlock::agetPosition;

}


//raw include
%include "limix/io/genotype.h"