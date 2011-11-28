/*
 * likelihood.cpp
 *
 *  Created on: Nov 11, 2011
 *      Author: clippert
 */

#include "likelihood.h"
#include "math.h"


namespace gpmix {

ALikelihood::ALikelihood(muint_t numberParams)
{
	this->numberParams=numberParams;
}

ALikelihood::~ALikelihood() {
	// TODO Auto-generated destructor stub
}

void ALikelihood::setParams(LikParams& params)
{
	if(this->numberParams!=(muint_t)params.rows())//WARNING: muint_t conversion
	{
		ostringstream os;
		os << "LikParams has wrong dimensions. params.rows() = "<< params.rows() << ", numberParams = "<< numberParams;
		throw gpmix::CGPMixException(os.str());
	}
}

CLikNormalIso::CLikNormalIso() : ALikelihood(1)
{
}

CLikNormalIso::~CLikNormalIso()
{
}


//CLikNormalIso

void CLikNormalIso::applyToK(MatrixXd& K) const
{
	if ((K.rows() != K.cols()))
	{
		ostringstream os;
		os << "K is not quadratic. K.rows() = "<< K.rows() << ", K.cols() = "<< K.cols();
		throw gpmix::CGPMixException(os.str());
	}

	mfloat_t sigma_2 = gpmix::exp( (mfloat_t)(2.0*this->getParams()(0)));//WARNING: mfloat_t conversion
	for (muint_t i = 0; i < (muint_t)K.rows(); ++i) //WARNING (muint_t) conversion
	{
		K(i,i) += sigma_2;
	}
}

void CLikNormalIso::K(MatrixXd* out) const
{
	mfloat_t sigma_2 = gpmix::exp( (mfloat_t)(2.0*this->getParams()(0)));//WARNING: mfloat_t conversion
	(*out) = MatrixXd::Zero(X.rows(),X.rows());
	for (muint_t i = 0; i < (muint_t)X.rows(); ++i)//WARNING: (muint_t) conversion
	{
		(*out)(i,i) = sigma_2;
	}
}

void CLikNormalIso::Kdiag(VectorXd* out) const
{
	mfloat_t sigma_2 = gpmix::exp( (mfloat_t)(2.0*this->getParams()(0)));//WARNING: mfloat_t conversion
	(*out).resize(X.rows());
	for (muint_t i = 0; i < (muint_t)X.rows(); ++i)//WARNING: (muint_t) conversion
	{
		(*out)(i) = sigma_2;
	}
}

void CLikNormalIso::Kgrad_params(MatrixXd* out, const muint_t row) const
{
	mfloat_t sigma_2 = 2.0*gpmix::exp( (mfloat_t)(2.0*this->getParams()(0)));//WARNING: mfloat_t conversion

	MatrixXd dK = MatrixXd::Zero(X.rows(),X.rows());
	for (muint_t i = 0; i < (muint_t)dK.rows(); ++i)//WARNING: (muint_t) conversion
	{
		(*out)(i,i) = sigma_2;
	}
}

} // end:: namespace gpmix

