/*
 * CLinearMean.cpp
 *
 *  Created on: Jan 3, 2012
 *      Author: clippert
 */

#include "CLinearMean.h"

namespace gpmix {
CLinearMean::CLinearMean() : ADataTerm::ADataTerm() {
	this->insync = false;
	this->nTargets = 0;
	this->fixedEffects = MatrixXd();
	this->weights = MatrixXd();
	this->Y = MatrixXd();
}

CLinearMean::CLinearMean(muint_t nTargets) : ADataTerm::ADataTerm() {
	this->insync = false;
	this->nTargets = nTargets;
	this->fixedEffects = MatrixXd();
	this->weights = MatrixXd();
	this->Y = MatrixXd();
}

CLinearMean::CLinearMean(MatrixXd& Y, MatrixXd& weights, MatrixXd& fixedEffects) : ADataTerm::ADataTerm(Y)
{
	this->insync = false;
	this->checkDimensions(Y, weights, fixedEffects, true, true, true);
	this->fixedEffects = fixedEffects;
	this->weights = weights;
}

CLinearMean::CLinearMean(MatrixXd& Y, MatrixXd& fixedEffects) : ADataTerm::ADataTerm(Y)
{
	this->checkDimensions(Y, weights, fixedEffects, true, false, true);
	this->insync = false;
	this->fixedEffects = fixedEffects;
	this->zeroInitWeights();
}

CLinearMean::~CLinearMean()
{
	// TODO Auto-generated destructor stub
}

void CLinearMean::aEvaluate(MatrixXd* outY)
{
	checkDimensions(weights,fixedEffects,Y, true, true, true);
	*outY = (this->Y - (this->fixedEffects * this->weights));
}

void CLinearMean::aGradParams(MatrixXd* outGradParams)
{
	this->checkDimensions(Y, *outGradParams, fixedEffects, true, true, true);
	*outGradParams = ( -fixedEffects );
}

void CLinearMean::zeroInitWeights()
{
	checkDimensions(MatrixXd(),this->Y,this->fixedEffects, false, true, true);
	this->insync = false;
	this->weights = MatrixXd::Zero(this->fixedEffects.cols(), this->Y.cols());
}

void CLinearMean::setParams(MatrixXd& weightMatrix)
{
	this->checkDimensions(weightMatrix, this->fixedEffects, this->Y, true, false, false);
	this->insync = false;
	this->weights = weightMatrix;
}

void CLinearMean::setFixedEffects(MatrixXd& fixedEffects)
{
	this->checkDimensions(this->weights, fixedEffects, this->Y, false, true, false);
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

void CLinearMean::aPredictY(MatrixXd* outY) const
{
	*outY = this->fixedEffects * this->weights;
}

void CLinearMean::aPredictYstar(MatrixXd* outY, const MatrixXd* fixedEffects) const
{
	*outY = (*fixedEffects) * this->weights;
}

void CLinearMean::setWeightsOLS()
{
	this->weights = this->fixedEffects.jacobiSvd().solve(this->Y);
}

void CLinearMean::setWeightsOLS(const MatrixXd& Y)
{
	this->weights = this->fixedEffects.jacobiSvd().solve(Y);
}

} /* namespace gpmix */
