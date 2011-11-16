/*
 * likelihood.cpp
 *
 *  Created on: Nov 11, 2011
 *      Author: clippert
 */

#include "likelihood.h"
#include "math.h"


namespace gpmix {

ALikelihood::ALikelihood() {
	// TODO Auto-generated constructor stub

}

ALikelihood::~ALikelihood() {
	// TODO Auto-generated destructor stub
}

CLikNormalIso::CLikNormalIso()
{
}

CLikNormalIso::~CLikNormalIso()
{
}

void CLikNormalIso::applyToK(const LikParams& params, MatrixXd& K) const
{
	if (params.rows()!=1 || params.cols()!=1)
	{
		ostringstream os;
		os << "LikParams is not a scalar. params.rows() = "<< params.rows() << ", params.cols() = "<< params.cols();
		throw gpmix::CGPMixException(os.str());
	}

	if (K.rows() != K.cols())
	{
		ostringstream os;
		os << "K is not quadratic. K.rows() = "<< K.rows() << ", K.cols() = "<< K.cols();
		throw gpmix::CGPMixException(os.str());
	}

	float_t sigma_2 = gpmix::exp( (float_t)(2.0*params(0,0)));//WARNING: float_t conversion
	for (uint_t i = 0; i < (uint_t)K.rows(); ++i) //WARNING (uint_t) conversion
	{
		K(i,i) += sigma_2;
	}
}

MatrixXd CLikNormalIso::K(const LikParams& params, MatrixXd& X) const
{
	float_t sigma_2 = gpmix::exp( (float_t)(2.0*params(0,0)));//WARNING: float_t conversion
	MatrixXd K = MatrixXd::Zero(X.rows(),X.rows());
	for (uint_t i = 0; i < (uint_t)K.rows(); ++i)//WARNING: (uint_t) conversion
	{
		K(i,i) = sigma_2;
	}
	return K;
}

VectorXd CLikNormalIso::Kdiag(const LikParams& params, MatrixXd& X) const
{
	float_t sigma_2 = gpmix::exp( (float_t)(2.0*params(0,0)));//WARNING: float_t conversion
	VectorXd Kdiag(X.rows());
	for (uint_t i = 0; i < (uint_t)Kdiag.rows(); ++i)//WARNING: (uint_t) conversion
	{
		Kdiag(i) = sigma_2;
	}
	return Kdiag;
}

MatrixXd CLikNormalIso::K_grad_theta(const LikParams& params, MatrixXd X, uint_t row) const
{
	if (params.rows()!=1 || params.cols()!=1 || row!=0)
		{
			ostringstream os;
			os << "LikParams is either not a scalar or the entry specified is not 0. params.rows() = "<< params.rows() << ", params.cols() = "<< params.cols()<<"row: "<<row;
			throw gpmix::CGPMixException(os.str());
		}
	float_t twoSigma_2 = 2.0*gpmix::exp( (float_t)(2.0*params(0,0)));//WARNING: float_t conversion

	MatrixXd dK = MatrixXd::Zero(X.rows(),X.rows());
	for (uint_t i = 0; i < (uint_t)dK.rows(); ++i)//WARNING: (uint_t) conversion
	{
		dK(i,i) = twoSigma_2;
	}
	return dK;
}

} // end:: namespace gpmix

