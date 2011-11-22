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
	// TODO Auto-generated destructor stub
}

MatrixXd CCovLinearISO::Kcross(const CovarInput& Xstar) const
{
	if (Xstar.rows()!=this->X.rows())
	{
		ostringstream os;
		os << this->getName() <<": Xstar has wrong number of dimensions. Xstar.cols() = "<< Xstar.cols() <<". X.cols() = "<< this->X.cols() << ".";
		throw gpmix::CGPMixException(os.str());
	}

	//kernel matrix is constant hyperparmeter and dot product
	float_t A = exp((float_t)(2.0*params(0)));
	return A* Xstar*this->X.transpose();
}


MatrixXd CCovLinearISO::K_grad_param( const uint_t i ) const
{
	if (i==0)
	{
		MatrixXd K = this->K();
		//devide by hyperparameter and we are done
		K*=2.0;
		return K;
	}
	else
	{
		ostringstream os;
		os << this->getName() <<": wrong index of hyperparameter. i = "<< i <<". this->params.cols() = "<< this->getNumberParams() << ".";
		throw gpmix::CGPMixException(os.str());
	}
}

MatrixXd CCovLinearISO::K_grad_X(const uint_t d) const
{
	float_t A = exp((float_t)(2.0*this->params(0)));
	//create empty matrix
	MatrixXd RV = MatrixXd::Zero(this->X.rows(),this->X.rows());
	//otherwise update computation:
	RV.colwise() = A*this->X.col(d);
	return RV;
}


MatrixXd CCovLinearISO::Kcross_grad_X(const CovarInput& Xstar, const uint_t d) const
{
	float_t A = exp((float_t)(2.0*this->params(0)));
	//create empty matrix
	MatrixXd RV = MatrixXd::Zero(Xstar.rows(),this->X.rows());
	//otherwise update computation:
	RV.colwise() = A*Xstar.col(d);
	return RV;
}

MatrixXd CCovLinearISO::Kdiag_grad_X( const uint_t d ) const
{
	float_t A = exp((float_t)(2.0*this->params(0)));
	VectorXd RV = VectorXd::Zero(this->X.rows());
	RV = 2.0*A*this->X.col(d);
	return RV;
}

VectorXd CCovLinearISO::Kdiag() const
{
	VectorXd RV = VectorXd::Zero(X.rows());
	return RV;
}


/***** CCovLinearARD *******/

CCovLinearARD::~CCovLinearARD() {
	// covaraince destructor
}

VectorXd CCovLinearARD::Kdiag() const
{
	VectorXd RV = VectorXd::Zero(X.rows());
	return RV;
}

MatrixXd CCovLinearARD::Kcross(const CovarInput& Xstar) const
{
	//kernel matrix is constant hyperparmeter and dot product

	//get all amplitude parameters, one per dimension
	VectorXd L = 2*params;
	L = L.unaryExpr(ptr_fun(exp));
	MatrixXd RV = Xstar*L.asDiagonal()*this->X.transpose();
	return RV;
}

MatrixXd CCovLinearARD::K_grad_param(const uint_t i) const
{
	//is the requested gradient within range?
	if (i >= (uint_t)this->X.cols()) //WARNING: uint_t conversion
		throw CGPMixException("unknown hyperparameter derivative requested in CLinearCFISO");
	//ok: calculcate:
	//1. get row i from x1:
	MatrixXd x1i = X.col(i);
	//2. get amplitude
	float_t A = exp((float_t)(2*params(i)));
	//outer product of the corresponding dimension.
	return A*2.0*(x1i*x1i.transpose());
}

MatrixXd CCovLinearARD::Kcross_grad_X(const CovarInput& Xstar, const uint_t d) const
{
	MatrixXd RV = MatrixXd::Zero(Xstar.rows(),X.rows());
	return RV;
}

MatrixXd CCovLinearARD::Kdiag_grad_X(const uint_t d) const
{
	VectorXd RV = VectorXd::Zero(X.rows());
	return RV;
}


} /* namespace gpmix */
