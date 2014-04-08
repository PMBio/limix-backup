// Copyright(c) 2014, The LIMIX developers (Christoph Lippert, Paolo Francesco Casale, Oliver Stegle)
// All rights reserved.
//
// LIMIX is provided under a 2-clause BSD license.
// See license.txt for the complete license.

#include "CData.h"

namespace limix {

CData::CData() : ADataTerm() {
}

CData::CData(MatrixXd& Y) : ADataTerm(Y) {
}

CData::~CData() {
}

} /* namespace limix */
