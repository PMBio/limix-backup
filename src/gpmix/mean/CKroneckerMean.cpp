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
	this->inSync = false;
	this->A = MatrixXd();
	this->fixedEffects = MatrixXd();
	this->weights = MatrixXd();
}

CKroneckerMean::~CKroneckerMean()
{

}

void CKroneckerMean::aGradParams(MatrixXd* out)
{
	//TODO How does this one look???
}

void CKroneckerMean::setWeightsOLS(const MatrixXd& Y)
{
	this->checkDimensions(Y, false);
	this->inSync = false;

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
	this->inSync = false;
	this->A = A;
}


} /* namespace gpmix */
