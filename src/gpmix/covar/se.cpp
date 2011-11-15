/*
 * CCovSqexpARD.cpp
 *
 *  Created on: Nov 11, 2011
 *      Author: stegle
 */

#include "se.h"
#include <math.h>
#include <cmath>
#include <gpmix/matrix/matrix_helper.h>
#include "dist.h"

namespace gpmix {


CCovSqexpARD::~CCovSqexpARD() {
	// TODO Auto-generated destructor stub
}


MatrixXd CCovSqexpARD::K(const CovarParams params, const CovarInput x1, const CovarInput x2) const
{
	CovarInput x1_ = this->getX(x1);
	CovarInput x2_ = this->getX(x2);
	//exponentiate hyperparams
	//amplitude
	float_t A = exp((float_t)(2.0*params(0)));
	//lengthscales
	VectorXd L = params.unaryExpr(ptr_fun(exp));
	//rescale with length
	MatrixXd x1l = L.asDiagonal().inverse() * x1;
	MatrixXd x2l = L.asDiagonal().inverse() * x1;
	//squared exponential distance
	MatrixXd RV = -0.5*sq_dist(x1l,x2l);
	RV = A*RV.unaryExpr(ptr_fun(exp));
	return RV;
} // end :: K


VectorXd CCovSqexpARD::Kdiag(const CovarParams params, const CovarInput x1) const
{
	float_t A = exp((float_t)(2.0*params(0)));
	return A*VectorXd::Ones(x1.rows());
}


MatrixXd CCovSqexpARD::Kgrad_theta(const CovarParams params, const CovarInput x1, const uint_t i) const
{
	CovarInput x1_ = this->getX(x1);
	MatrixXd RV = K(params,x1,x1);
	if (i==0)
		//derivative w.r.t. amplitude
		RV*=2.0;
	else if ((i>0) && (i<this->dimensions_i1))
	{
		//lengthscale derivative:
		//1. get row i from x1:
		MatrixXd x1i = x1.row(i);
		//reweight with lengthscale
		x1i.array()/= exp((float_t)params(i));
		//2. calculate all squqared distance
		MatrixXd sq = sq_dist(x1i,x1i);
		//3. elementwise product
		RV*=sq;
	}
	return RV;
}

MatrixXd CCovSqexpARD::Kgrad_x(const CovarParams params, const CovarInput x1, const CovarInput x2, const uint_t d) const
{
	return MatrixXd::Zero(x1.rows(),x2.rows());
}

MatrixXd CCovSqexpARD::Kgrad_xdiag(const CovarParams params, const CovarInput x1, const uint_t d) const
{
	return VectorXd(x1.rows());
}







} /* namespace gpmix */
