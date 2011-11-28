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


void CCovSqexpARD::Kcross(MatrixXd* out, const CovarInput& Xstar ) const
{
	//amplitude
	mfloat_t A = exp((mfloat_t)(2.0*params(0)));
	//lengthscales
	MatrixXd L = params.block(1,0,params.rows()-1,1).unaryExpr(ptr_fun(exp));
	//rescale with length
	MatrixXd x1l = Xstar * L.asDiagonal().inverse();
	MatrixXd x2l = this->X * L.asDiagonal().inverse();
	//squared exponential distance
	MatrixXd RV = -0.5*sq_dist(x1l,x2l);
	(*out) = A*RV.unaryExpr(ptr_fun(exp));
} // end :: K

void CCovSqexpARD::Kgrad_param(MatrixXd* out,const muint_t i) const
{
	(*out) = K();
	if (i==0)
		//derivative w.r.t. amplitude
		(*out)*=2.0;
	else if ((i>0) && (i<getNumberParams()))
	{
		//lengthscale derivative:
		//1. get row i from x1:
		MatrixXd x1i = this->X.row(i);
		//reweight with lengthscale
		x1i.array()/= exp((mfloat_t)params(i));
		//2. calculate all squqared distance
		MatrixXd sq = sq_dist(x1i,x1i);
		//3. elementwise product
		(*out)*=sq;
	}
}

void CCovSqexpARD::Kcross_grad_X(MatrixXd* out,const CovarInput& Xstar, const muint_t d) const
{
}

void CCovSqexpARD::Kdiag_grad_X(MatrixXd* out,const muint_t d) const
{
}




} /* namespace gpmix */
