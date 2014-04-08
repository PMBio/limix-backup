// Copyright(c) 2014, The LIMIX developers (Christoph Lippert, Paolo Francesco Casale, Oliver Stegle)
// All rights reserved.
//
// LIMIX is provided under a 2-clause BSD license.
// See license.txt for the complete license.

#include "CKroneckerMean.h"

namespace limix {


CKroneckerMean::CKroneckerMean()
{
}

CKroneckerMean::~CKroneckerMean()
{

}

CKroneckerMean::CKroneckerMean(MatrixXd& Y, MatrixXd& weights, MatrixXd& fixedEffects, MatrixXd& A) : CLinearMean(Y,weights,fixedEffects)
{
	this->A=A;
}

CKroneckerMean::CKroneckerMean(MatrixXd& Y, MatrixXd& fixedEffects, MatrixXd& A) : CLinearMean(Y,fixedEffects)
{
	this->A=A;
}

void CKroneckerMean::aPredictY(MatrixXd* outY) const
{
	*outY = this->fixedEffects * this->weights * this->A;
}

void CKroneckerMean::aEvaluate(MatrixXd* outY)
{
	//checkDimensions(weights,fixedEffects,Y, true, true, true);
	*outY = (this->Y - (this->fixedEffects * this->weights) * this->A);
}

void CKroneckerMean::aGradParams(MatrixXd* outGradParams, const MatrixXd* KinvY)
{
	//std::cout << this->weights << "\n";
	*outGradParams = ( -fixedEffects ).transpose() * (*KinvY) * this->A.transpose();
}

void CKroneckerMean::setWeightsOLS(const MatrixXd& Y)
{
	this->checkDimensions(Y, false);
	//TODO: make case nTargets > nSamples efficient
	MatrixXd Adagger;
	if (A.rows() < A.cols())
	{
		Adagger = A.transpose() * (this->A * this->A.transpose()).inverse();
	}
	else
	{
		Adagger = (this->A.transpose() * this->A).inverse() * this->A;
	}
	MatrixXd YAd = Y*Adagger;
	this->weights = this->fixedEffects.jacobiSvd().solve(YAd);
	propagateSync(false);
}

void CKroneckerMean::setA(const MatrixXd& A)
{
	this->A = A;
	propagateSync(false);
}


} /* namespace limix */
