/*
 * ACovariance.cpp
 *
 *  Created on: Nov 10, 2011
 *      Author: stegle
 */

#include "covariance.h"

namespace gpmix {

ACovarianceFunction::ACovarianceFunction(const uint_t dimensions, const uint_t dimensions_i0, const uint_t dimensions_i1)
{
	//do we have an explicit dimension mask as parameter
	if((dimensions_i1>0) & (dimensions_i0>0))
	{
		this->dimensions_i0 = dimensions_i0;
		this->dimensions_i1 = dimensions_i1;
	}
	//otherwise number of data dimensions at least?
	else if(dimensions>0)
	{
		this->dimensions = dimensions;
		this->dimensions_i0 = 0;
		this->dimensions_i1 = this->dimensions;
	}
	//ok; then we are in trouble
	//TODO throw exception

}

ACovarianceFunction::~ACovarianceFunction()
{
	// TODO Auto-generated destructor stub
}

CovarInput ACovarianceFunction::getX(CovarInput x)
{
	return x.block(0,dimensions_i0,x.rows(),dimensions_i1);
}

bool ACovarianceFunction::dimension_is_target(const uint_t d)
{
	return ((d>=this->dimensions_i0) & (d<this->dimensions_i1));
}

VectorXd ACovarianceFunction::Kdiag(CovarParams params,CovarInput x1)
/*
 * Default implementation of diagional covariance operator
 */
{
	MatrixXd K = this->K(params,x1,x1);
	return K.diagonal();
}


} /* namespace gpmix */
