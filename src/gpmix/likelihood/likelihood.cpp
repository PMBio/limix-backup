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

void CLikNormalIso::applyToK(const MatrixXd& X, MatrixXd& K) const
{

	if ((K.rows() != K.cols()) || (K.cols()!=X.rows()))
	{
		ostringstream os;
		os << "K is not quadratic. K.rows() = "<< K.rows() << ", K.cols() = "<< K.cols()<<"X.rows() = " << X.rows();
		throw gpmix::CGPMixException(os.str());
	}

	mfloat_t sigma_2 = gpmix::exp( (mfloat_t)(2.0*this->getParams()(0)));//WARNING: mfloat_t conversion
	for (muint_t i = 0; i < (muint_t)K.rows(); ++i) //WARNING (muint_t) conversion
	{
		K(i,i) += sigma_2;
	}
}

MatrixXd CLikNormalIso::K(const MatrixXd& X) const
{
	mfloat_t sigma_2 = gpmix::exp( (mfloat_t)(2.0*this->getParams()(0)));//WARNING: mfloat_t conversion
	MatrixXd K = MatrixXd::Zero(X.rows(),X.rows());
	for (muint_t i = 0; i < (muint_t)K.rows(); ++i)//WARNING: (muint_t) conversion
	{
		K(i,i) = sigma_2;
	}
	return K;
}

VectorXd CLikNormalIso::Kdiag(const MatrixXd& X) const
{
	mfloat_t sigma_2 = gpmix::exp( (mfloat_t)(2.0*this->getParams()(0)));//WARNING: mfloat_t conversion
	VectorXd Kdiag(X.rows());
	for (muint_t i = 0; i < (muint_t)Kdiag.rows(); ++i)//WARNING: (muint_t) conversion
	{
		Kdiag(i) = sigma_2;
	}
	return Kdiag;
}

MatrixXd CLikNormalIso::K_grad_params(const MatrixXd& X, const muint_t row) const
{
	mfloat_t twoSigma_2 = 2.0*gpmix::exp( (mfloat_t)(2.0*this->getParams()(0)));//WARNING: mfloat_t conversion

	MatrixXd dK = MatrixXd::Zero(X.rows(),X.rows());
	for (muint_t i = 0; i < (muint_t)dK.rows(); ++i)//WARNING: (muint_t) conversion
	{
		dK(i,i) = twoSigma_2;
	}
	return dK;
}

} // end:: namespace gpmix

