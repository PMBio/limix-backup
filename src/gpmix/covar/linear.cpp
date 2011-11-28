/*
 * Linear.cpp
 *
 *  Created on: Nov 11, 2011
 *      Author: stegle
 */

#include "linear.h"
#include <math.h>
#include <cmath>
#include <gpmix/utils/matrix_helper.h>

namespace gpmix {


/***** CCovLinearISO *******/

CCovLinearISO::~CCovLinearISO() {
}

void CCovLinearISO::Kcross(MatrixXd* out,const CovarInput& Xstar) const
{
	//create result matrix:
	out->resize(this->X.rows(),Xstar.rows());
	//We dfine Xstar [N X D] where N are samples...
	if (Xstar.cols()!=this->X.cols())
	{
		ostringstream os;
		os << this->getName() <<": Xstar has wrong number of dimensions. Xstar.cols() = "<< Xstar.cols() <<". X.cols() = "<< this->X.cols() << ".";
		throw gpmix::CGPMixException(os.str());
	}
	//kernel matrix is constant hyperparmeter and dot product
	mfloat_t A = exp((mfloat_t)(2.0*params(0)));
	(*out) = A* Xstar*this->X.transpose();
}

void CCovLinearISO::Kgrad_param(MatrixXd* out, const muint_t i ) const
{
	if (i==0)
	{
		out->resize(this->X.rows(),this->X.rows());
		(*out) = 2.0*this->K();
	}
	else
	{
		ostringstream os;
		os << this->getName() <<": wrong index of hyperparameter. i = "<< i <<". this->params.cols() = "<< this->getNumberParams() << ".";
		throw gpmix::CGPMixException(os.str());
	}
}

void CCovLinearISO::Kcross_grad_X(MatrixXd* out,const CovarInput& Xstar, const muint_t d) const
{
	mfloat_t A = exp((mfloat_t)(2.0*this->params(0)));
	//create empty matrix
	(*out) = MatrixXd::Zero(Xstar.rows(),this->X.rows());
	//otherwise update computation:
	(*out).rowwise() = A*Xstar.col(d);
}

void CCovLinearISO::Kdiag_grad_X(VectorXd* out, const muint_t d ) const
{
	mfloat_t A = exp((mfloat_t)(2.0*this->params(0)));
	(*out) = VectorXd::Zero(this->X.rows());
	(*out) = 2.0*A*this->X.col(d);
}


/***** CCovLinearARD *******/

CCovLinearARD::~CCovLinearARD()
{
}


//overloaded pure virtual functions:
void CCovLinearARD::Kcross(MatrixXd* out, const CovarInput& Xstar ) const
{
	//get all amplitude parameters, one per dimension
	VectorXd L = 2*params;
	L = L.unaryExpr(ptr_fun(exp));
	(*out) = Xstar*L.asDiagonal()*this->X.transpose();
}

void CCovLinearARD::Kgrad_param(MatrixXd* out,const muint_t i) const
{
	//is the requested gradient within range?
	if (i >= (muint_t)this->X.cols()) //WARNING: muint_t conversion
		throw CGPMixException("unknown hyperparameter derivative requested in CLinearCFISO");
	//ok: calculcate:
	//1. get row i from x1:
	MatrixXd x1i = X.col(i);
	//2. get amplitude
	mfloat_t A = exp((mfloat_t)(2*params(i)));
	//outer product of the corresponding dimension.
	(*out) =  A*2.0*(x1i*x1i.transpose());
}

void CCovLinearARD::Kcross_grad_X(MatrixXd* out,const CovarInput& Xstar, const muint_t d) const
{
}

void CCovLinearARD::Kdiag_grad_X(VectorXd* out,const muint_t d) const
{
}




} /* namespace gpmix */
