// Copyright(c) 2014, The LIMIX developers (Christoph Lippert, Paolo Francesco Casale, Oliver Stegle)
// All rights reserved.
//
// LIMIX is provided under a 2-clause BSD license.
// See license.txt for the complete license.

#include "CLinearMean.h"

namespace limix {
CLinearMean::CLinearMean() : ADataTerm()
{
	this->nTargets = 0;
}

CLinearMean::CLinearMean(muint_t nTargets) : ADataTerm()
{
	this->nTargets = nTargets;
}

CLinearMean::CLinearMean(const MatrixXd& Y, const MatrixXd& weights, const MatrixXd& fixedEffects) : ADataTerm(Y)
{
	this->checkDimensions(weights, fixedEffects, Y, true, true, true);
	this->fixedEffects = fixedEffects;
	this->weights = weights;
	this->nTargets = Y.cols();
}

CLinearMean::CLinearMean(const MatrixXd& Y, const MatrixXd& fixedEffects) : ADataTerm(Y)
{
	this->checkDimensions(weights, fixedEffects, Y, true, false, true);
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
	*outY = this->Y;
	if (!isnull(this->weights))
		*outY -= (this->fixedEffects * this->weights);
}

void CLinearMean::aGradParams(MatrixXd* outGradParams, const MatrixXd* KinvY)
{
	this->checkDimensions(Y, *outGradParams, fixedEffects, true, true, true);
	*outGradParams = ( -fixedEffects ).transpose() * (*KinvY);
}

void CLinearMean::zeroInitWeights()
{
	checkDimensions(MatrixXd(),this->Y,this->fixedEffects, false, true, true);
	this->weights = MatrixXd::Zero(this->fixedEffects.cols(), this->Y.cols());
	propagateSync(false);
}

void CLinearMean::setParams(const MatrixXd& weightMatrix)
{
	this->checkDimensions(weightMatrix, this->fixedEffects, this->Y, true, false, false);
	this->weights = weightMatrix;
	propagateSync(false);
}

void CLinearMean::setFixedEffects(const MatrixXd& fixedEffects)
{
	this->checkDimensions(this->weights, fixedEffects, this->Y, false, true, false);
	propagateSync(false);
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
	propagateSync(false);
}

void CLinearMean::setWeightsOLS(const MatrixXd& Y)
{
	this->weights = this->fixedEffects.jacobiSvd().solve(Y);
	propagateSync(false);
}

} /* namespace limix */
