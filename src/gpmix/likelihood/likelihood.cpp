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
	for (int_t i = 0; i < K.rows(); ++i)
	{
		K(i,i) += sigma_2;
	}
}


} // end:: namespace gpmix

