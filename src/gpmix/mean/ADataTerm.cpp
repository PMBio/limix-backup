/*
 * ADataTerm.cpp
 *
 *  Created on: Jan 3, 2012
 *      Author: clippert
 */

#include "ADataTerm.h"

namespace gpmix {
ADataTerm::ADataTerm()
{
}

ADataTerm::ADataTerm(MatrixXd& Y) {
	this->Y = Y;
}

ADataTerm::~ADataTerm()
{
}

bool ADataTerm::isInSync() const
{return insync;}

void ADataTerm::makeSync()
{ insync = true;}

} /* namespace gpmix */
