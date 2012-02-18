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
	insync = false;
}

ADataTerm::ADataTerm(MatrixXd& Y)
{
	this->Y = Y;
	insync = false;
}

ADataTerm::~ADataTerm()
{
}

bool ADataTerm::isInSync() const
{return insync;}

void ADataTerm::makeSync()
{ insync = true;}

void ADataTerm::aEvaluate(MatrixXd* outY)
{
	*outY = this->Y;
}

void ADataTerm::aGradY(MatrixXd* outGradY)
{
	*outGradY = MatrixXd::Ones(this->Y.rows(), this->Y.cols());
}

void ADataTerm::aGradParams(MatrixXd* outGradParams, const MatrixXd* KinvY)
{
	*outGradParams = MatrixXd();
}

void ADataTerm::aSumJacobianGradParams(MatrixXd* outSumJacobianGradParams)
{
	*outSumJacobianGradParams = MatrixXd();
}

void ADataTerm::aSumLogJacobian(MatrixXd* outSumJacobianGradParams)
{
	*outSumJacobianGradParams = MatrixXd();
}



} /* namespace gpmix */
