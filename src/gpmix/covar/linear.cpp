/*
 * Linear.cpp
 *
 *  Created on: Nov 11, 2011
 *      Author: stegle
 */

#include "linear.h"

namespace gpmix {

CLinearCFISO::CLinearCFISO() {
	// TODO Auto-generated constructor stub

}

CLinearCFISO::~CLinearCFISO() {
	// TODO Auto-generated destructor stub
}


MatrixXd CLinearCFISO::K(CovarParams params,CovarInput x1,CovarInput x2)
{
	//kernel matrix is constant hyperparmeter and dot product
	CovarInput x1_ = this->getX(x1);
	CovarInput x2_ = this->getX(x2);

	//TODO: discuss whether to exponentiate parameters or not.
	double A = params(0);
	return A* x1*x2.transpose();
}


MatrixXd CLinearCFISO::Kgrad_theta(CovarParams params, CovarInput x1,int i)
{
	if (i==0)
	{
		MatrixXd K = this->K(params,x1,x1);
		//device by hyperparameter and we are done
		K/= params(0);
		return K;
	}
	else
	{
		//TODO throw exception.
		return MatrixXd(1,1);
	}
}


MatrixXd CLinearCFISO::Kgrad_x(CovarParams params,CovarInput x1,CovarInput x2,int d)
{
	double A = params(0);
	//check that the requested dimension is actually a target of this covariance
	if (not this->dimension_is_target(d))
	{
		return MatrixXd::Zero(x1.rows(),x2.rows());
	}
	//ok, if it is we have to do some work
	return A*x2.col(d);
	//TODO: check how this is actually done in pygp... not sure anymore
}

MatrixXd CLinearCFISO::Kgrad_xdiag(CovarParams params,CovarInput x1,int d)
{
	double A = params(0);
	if (not this->dimension_is_target(d))
		{
			return MatrixXd::Zero(x1.rows(),x1.rows());
		}
	return A*2*x1.col(d);
}



} /* namespace gpmix */
