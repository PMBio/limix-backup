/*
 * CKroneckerMean.cpp
 *
 *  Created on: Jan 19, 2012
 *      Author: clippert
 */

#include "CKroneckerMean.h"

namespace gpmix {


CKroneckerMean::CKroneckerMean(muint_t nSamples, muint_t nTargets)
{
	this->insync = false;
	this->A = MatrixXd();
	this->fixedEffects = MatrixXd();
	this->weights = MatrixXd();
}

CKroneckerMean::~CKroneckerMean()
{

}

CKroneckerMean::CKroneckerMean(MatrixXd& Y, MatrixXd& weights, MatrixXd& fixedEffects, MatrixXd& A) : CLinearMean(Y,weights,fixedEffects)
{
	this->A=A;
}

void CKroneckerMean::aPredictY(MatrixXd* outY) const
{
	*outY = this->fixedEffects * this->weights * this->A;
}

void CKroneckerMean::aEvaluate(MatrixXd* outY)
{
	checkDimensions(weights,fixedEffects,Y, true, true, true);
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
	this->insync = false;

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
}

void CKroneckerMean::setA(MatrixXd& A)
{
	this->checkDimensions(fixedEffects, weights, A);
	this->insync = false;
	this->A = A;
}


} /* namespace gpmix */
