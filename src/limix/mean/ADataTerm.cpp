/*
 * ADataTerm.cpp
 *
 *  Created on: Jan 3, 2012
 *      Author: clippert
 */

#include "ADataTerm.h"

namespace limix {
ADataTerm::ADataTerm()
{
}

ADataTerm::ADataTerm(const MatrixXd& Y)
{
	this->Y = Y;
}

ADataTerm::~ADataTerm()
{
}

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



} /* namespace limix */
