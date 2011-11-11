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
	if (K.rows() != K.cols())
	{
		printf("\nK is not symmetric!");
		throw 1;
	}

	float_t sigma_2 = (float_t)std::exp( (long double)(2.0*params(0,0)));
	for (int_t i = 0; i < K.rows(); ++i)
	{
		K(i,i) += sigma_2;
	}
}


} // end:: namespace gpmix

