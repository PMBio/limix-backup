// Copyright(c) 2014, The LIMIX developers(Christoph Lippert, Paolo Francesco Casale, Oliver Stegle)
//
//Licensed under the Apache License, Version 2.0 (the "License");
//you may not use this file except in compliance with the License.
//You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
//Unless required by applicable law or agreed to in writing, software
//distributed under the License is distributed on an "AS IS" BASIS,
//WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//See the License for the specific language governing permissions and
//limitations under the License.

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
