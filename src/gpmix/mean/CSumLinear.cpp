/*
 * CSumLinear.cpp
 *
 *  Created on: Jan 23, 2012
 *      Author: clippert
 */

#include "CSumLinear.h"

namespace gpmix {

CSumLinear::CSumLinear() {

}

CSumLinear::~CSumLinear() {

}

void CSumLinear::aGetParams(MatrixXd* outParams)
{
	muint_t sumParams = 0;
	for (muint_t i = 0; i<this->terms.size(); ++i)
	{
		sumParams += this->terms[i]->getRowsParams() * this->terms[i]->getColsParams();
	}
	outParams->resize(sumParams, 1);
	sumParams = 0;
	for (muint_t i = 0; i<this->terms.size(); ++i)
	{
		MatrixXd currentParams = terms[i]->getParams();
		outParams->block(sumParams,0,this->terms[i]->getRowsParams() * this->terms[i]->getColsParams(),1).array() = currentParams.array();
		sumParams += this->terms[i]->getRowsParams() * this->terms[i]->getColsParams();
	}
}

void CSumLinear::setParams(const MatrixXd& params)
{
	muint_t sumParams = 0;
	for (muint_t i = 0; i<this->terms.size(); ++i)
	{
		MatrixXd currentGradParams = MatrixXd(this->terms[i]->getRowsParams() * this->terms[i]->getColsParams(),1);
		currentGradParams.array() = params.block(sumParams,0,currentGradParams.rows(),1).array();
		currentGradParams.resize(this->terms[i]->getRowsParams() , this->terms[i]->getColsParams());
		this->terms[i]->setParams(currentGradParams);
		sumParams += this->terms[i]->getRowsParams() * this->terms[i]->getColsParams();
	}
}

void CSumLinear::aEvaluate(MatrixXd* Y)
{
	(*Y) = this->Y;
	MatrixXd pred;

	//collect predicitons from each subterm
	for (size_t i=0; i<this->terms.size(); ++i)
	{
		this->terms[i]->aPredictY(&pred);
		*Y -= pred;
	}
}

void CSumLinear::aGradParams(MatrixXd* outGradParams, const MatrixXd* KinvY)
{
	muint_t sumParams = 0;
	for (muint_t i = 0; i<this->terms.size(); ++i)
	{
		sumParams += this->terms[i]->getRowsParams() * this->terms[i]->getColsParams();
	}
	outGradParams->resize(sumParams, 1);
	sumParams = 0;
	for (muint_t i = 0; i<this->terms.size(); ++i)
	{
		MatrixXd currentGradParams = terms[i]->gradParams(*KinvY);
		currentGradParams.resize(this->terms[i]->getRowsParams() * this->terms[i]->getColsParams(),1);
		outGradParams->block(sumParams,0,currentGradParams.rows(),1).array() = currentGradParams.array();
		sumParams += this->terms[i]->getRowsParams() * this->terms[i]->getColsParams();
	}
}

} /* namespace gpmix */
