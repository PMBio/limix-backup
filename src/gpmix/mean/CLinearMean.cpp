/*
 * CLinearMean.cpp
 *
 *  Created on: Jan 3, 2012
 *      Author: clippert
 */

#include "CLinearMean.h"

namespace gpmix {

CLinearMean::CLinearMean() : ADataTerm::ADataTerm() {
	// TODO Auto-generated constructor stub
	this->insync = false;
}

CLinearMean::CLinearMean(MatrixXd& Y, MatrixXd& weights, MatrixXd& fixedEffects) : ADataTerm::ADataTerm(Y)
{
	this->insync = false;
	this->fixedEffects = fixedEffects;
	this->weights = weights;
}

CLinearMean::CLinearMean(MatrixXd& Y, MatrixXd& fixedEffects) : ADataTerm::ADataTerm(Y)
{
	this->insync = false;
	this->zeroInitWeights();
}

CLinearMean::~CLinearMean()
{
	// TODO Auto-generated destructor stub
}

void CLinearMean::aEvaluate(MatrixXd* outY)
{
	*outY = (this->Y - (this->fixedEffects * this->weights));
}

void CLinearMean::aGradParams(MatrixXd* outGradParams)
{
	*outGradParams = ( -fixedEffects );
}

void CLinearMean::zeroInitWeights()
{
	this->insync = false;
	this->weights = MatrixXd::Zero(this->fixedEffects.cols(), this->Y.cols());
}

void CLinearMean::setParams(MatrixXd& weightMatrix)
{
	this->insync = false;
	this->weights = weightMatrix;
}

void CLinearMean::setfixedEffects(MatrixXd& fixedEfects)
{
	this->insync = false;
	this->zeroInitWeights();
	this->fixedEffects = fixedEffects;
}

void CLinearMean::aGetParams(MatrixXd* outParams)
{
	*outParams = this->weights;
}

void CLinearMean::aGetFixedEffects(MatrixXd* outFixedEffects)
{
	*outFixedEffects = this->fixedEffects;
}

} /* namespace gpmix */
