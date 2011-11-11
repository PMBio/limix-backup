/*
 * Linear.cpp
 *
 *  Created on: Nov 11, 2011
 *      Author: stegle
 */

#include "linear.h"
#include <math.h>
#include <cmath>
#include <gpmix/matrix/matrix_helper.h>

namespace gpmix {

CCovLinearISO::~CCovLinearISO() {
	// TODO Auto-generated destructor stub
}

MatrixXd CCovLinearISO::K(const CovarParams params, const CovarInput x1, const CovarInput x2) const
{
	//kernel matrix is constant hyperparmeter and dot product
	CovarInput x1_ = this->getX(x1);
	CovarInput x2_ = this->getX(x2);

	float_t A = exp((float_t)(2.0*params(0)));
	return A* x1*x2.transpose();
}


MatrixXd CCovLinearISO::Kgrad_theta(const CovarParams params, const CovarInput x1, const uint_t i) const
{
	if (i==0)
	{
		MatrixXd K = this->K(params,x1,x1);
		//device by hyperparameter and we are done
		K*=2.0;
		return K;
	}
	else
		throw CGPMixException("unknown hyperparameter derivative requested in CLinearCFISO");
}


MatrixXd CCovLinearISO::Kgrad_x(const CovarParams params, const CovarInput x1, const CovarInput x2, const uint_t d) const
{
	float_t A = exp((float_t)(2.0*params(0)));
	//create empty matrix
	MatrixXd RV = MatrixXd::Zero(x1.rows(),x2.rows());
	//check that the requested dimension is actually a target of this covariance
	if (not this->dimension_is_target(d))
		return RV;
	//otherwise update computation:
	RV.rowwise() = A*x2.col(d);
	return RV;
}

MatrixXd CCovLinearISO::Kgrad_xdiag(const CovarParams params, const CovarInput x1, const uint_t d) const
{
	float_t A = exp((float_t)(2.0*params(0)));
	VectorXd RV = VectorXd::Zero(x1.rows());
	if (not this->dimension_is_target(d))
		return MatrixXd::Zero(x1.rows(),x1.rows());
	RV = 2.0*A*x1.col(d);
	return RV;
}

} /* namespace gpmix */
