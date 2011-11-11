/*
 * ACovariance.cpp
 *
 *  Created on: Nov 10, 2011
 *      Author: stegle
 */

#include "covariance.h"

namespace gpmix {

ACovarianceFunction::ACovarianceFunction(const uint_t dimensions)
{
	this->dimensions_i1 = dimensions;
	this->dimensions_i0 = 0;
}

ACovarianceFunction::ACovarianceFunction(const uint_t dimensions_i0,const uint_t dimensions_i1)
{
	this->dimensions_i0 = dimensions_i0;
	this->dimensions_i1 = dimensions_i1;
}

ACovarianceFunction::~ACovarianceFunction()
{
}

CovarInput ACovarianceFunction::getX(const CovarInput x) const
{
	return x.block(0,dimensions_i0,x.rows(),dimensions_i1);
}

bool ACovarianceFunction::dimension_is_target(const uint_t d) const
{
	return ((d>=this->dimensions_i0) & (d<this->dimensions_i1));
}

VectorXd ACovarianceFunction::Kdiag(CovarParams params,CovarInput x1) const
/*
 * Default implementation of diagional covariance operator
 */
{
	MatrixXd K = this->K(params,x1,x1);
	return K.diagonal();
}


} /* namespace gpmix */
