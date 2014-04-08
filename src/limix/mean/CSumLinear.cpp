// Copyright(c) 2014, The LIMIX developers (Christoph Lippert, Paolo Francesco Casale, Oliver Stegle)
// All rights reserved.
//
// LIMIX is provided under a 2-clause BSD license.
// See license.txt for the complete license.

#include "CSumLinear.h"

namespace limix {

CSumLinear::CSumLinear() {

}

CSumLinear::~CSumLinear() {

}

void CSumLinear::aGetParams(MatrixXd* outParams)
{
	muint_t sumParams = getRowsParams()*getColsParams();
	outParams->resize(sumParams, 1);

	sumParams = 0;
	for(VecLinearMean::const_iterator iter = terms.begin(); iter!=terms.end();iter++)
	{
		MatrixXd currentParams = iter[0]->getParams();
		outParams->block(sumParams,0,iter[0]->getRowsParams() * iter[0]->getColsParams(),1).array() = currentParams.array().transpose();
		sumParams += iter[0]->getRowsParams() * iter[0]->getColsParams();
	}
}

void CSumLinear::setParams(const MatrixXd& params)
{
	muint_t sumParams = 0;
	for(VecLinearMean::const_iterator iter = terms.begin(); iter!=terms.end();iter++)
	{
		MatrixXd currentGradParams = MatrixXd(iter[0]->getRowsParams() * iter[0]->getColsParams(),1);
		currentGradParams.array() = params.block(sumParams,0,currentGradParams.rows(),1).array();
		currentGradParams.resize(iter[0]->getRowsParams() , iter[0]->getColsParams());
		iter[0]->setParams(currentGradParams);
		sumParams += iter[0]->getRowsParams() * iter[0]->getColsParams();
	}
	propagateSync(false);
}

muint_t CSumLinear::getRowsParams()
{
	muint_t sumParams = 0;
	for(VecLinearMean::const_iterator iter = terms.begin(); iter!=terms.end();iter++)
	{
		sumParams+=iter[0]->getRowsParams()*iter[0]->getColsParams();
	}
	return sumParams;
}
muint_t CSumLinear::getColsParams()
{
	return 1;
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

} /* namespace limix */
