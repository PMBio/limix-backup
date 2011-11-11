/*
 * fixed.cpp
 *
 *  Created on: Nov 11, 2011
 *      Author: stegle
 */

#include "fixed.h"

namespace gpmix {

CFixedCF::CFixedCF(MatrixXd K0)
{
	this->K0 = K0;
}

CFixedCF::~CFixedCF() {
	// TODO Auto-generated destructor stub
}

} /* namespace gpmix */
