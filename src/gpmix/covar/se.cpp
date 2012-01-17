/*
 * CCovSqexpARD.cpp
 *
 *  Created on: Nov 11, 2011
 *      Author: stegle
 */
#include "se.h"
#include <math.h>
#include <cmath>
#include <gpmix/utils/matrix_helper.h>
#include "dist.h"

namespace gpmix {


CCovSqexpARD::~CCovSqexpARD() {
	// TODO Auto-generated destructor stub
}


void CCovSqexpARD::setNumberDimensions(muint_t numberDimensions)
{
	this->numberDimensions = numberDimensions;
	this->numberParams = numberDimensions+1;
}


void CCovSqexpARD::aKcross(MatrixXd* out, const CovarInput& Xstar ) const throw (CGPMixException)
{
	//amplitude
	mfloat_t A = exp((mfloat_t)(2.0*params(0)));
	//lengthscales
	MatrixXd L = params.block(1,0,params.rows()-1,1).unaryExpr(ptr_fun(exp));
	//rescale with length
	MatrixXd x1l = Xstar * L.asDiagonal().inverse();
	MatrixXd x2l = this->X * L.asDiagonal().inverse();
	//squared exponential distance
	MatrixXd RV;
	sq_dist(&RV,x1l,x2l);
	RV*= -0.5;
	(*out) = A*RV.unaryExpr(ptr_fun(exp));
} // end :: K

void CCovSqexpARD::aKcross_diag(VectorXd* out, const CovarInput& Xstar) const throw(CGPMixException)
		{
		mfloat_t A = exp((mfloat_t)(2.0*params(0)));
		(*out) = A*VectorXd::Ones(Xstar.rows());
		}


void CCovSqexpARD::aKgrad_param(MatrixXd* out,const muint_t i) const throw (CGPMixException)
{
	//code copied from K
	mfloat_t A = exp((mfloat_t)(2.0*params(0)));
	//lengthscales
	MatrixXd L = params.block(1,0,params.rows()-1,1).unaryExpr(ptr_fun(exp));
	//rescale with length
	MatrixXd x1l = X * L.asDiagonal().inverse();
	//squared exponential distance
	MatrixXd RV;
	sq_dist(&RV,x1l,x1l);
	RV*= -0.5;
	(*out) = A*RV.unaryExpr(ptr_fun(exp));

	if (i==0)
		//derivative w.r.t. amplitude
		(*out)*=2.0;
	else if ((i>0) && (i<getNumberParams()))
	{
		//lengthscale derivative:
		//1. get col (i-1) from X:
		MatrixXd sq;
		sq_dist(&sq,x1l.col(i-1),x1l.col(i-1));
		//3. elementwise product
		(*out).array()*=sq.array();
	}
	else
	{
		throw CGPMixException("Parameter outside range");
	}
}

void CCovSqexpARD::aKcross_grad_X(MatrixXd* out,const CovarInput& Xstar, const muint_t d) const throw (CGPMixException)
{
	this->aKcross(out,Xstar);
	//lengthscales: now we need to squre explicitly
	MatrixXd L2 = 2.0*params.block(1,0,params.rows()-1,1);
	L2 = L2.unaryExpr(ptr_fun(exp));
	MatrixXd dist;
	lin_dist(&dist,X,Xstar,d);
	//rescale with squared lengthscale of the corresponding dimension:
	dist/=(-1.0*L2(d));
	//pointwise product with out array
	(*out).array() *= dist.array();
}

void CCovSqexpARD::aKdiag_grad_X(VectorXd* out,const muint_t d) const throw (CGPMixException)
{
	(*out) = VectorXd::Zero(X.rows());
}




} /* namespace gpmix */
