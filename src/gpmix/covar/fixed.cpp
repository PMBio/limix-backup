/*
 * fixed.cpp
 *
 *  Created on: Nov 11, 2011
 *      Author: stegle
 */
#include "fixed.h"
#include "gpmix/types.h"
#include "assert.h"

namespace gpmix {



gpmix::CFixedCF::CFixedCF(const MatrixXd & K0) : ACovarianceFunction(1)
{
	this->K0 = K0;
}


gpmix::CFixedCF::~CFixedCF()
{
}


void gpmix::CFixedCF::aKcross(MatrixXd *out, const CovarInput & Xstar) const throw(CGPMixException)
{
	mfloat_t A = exp((mfloat_t)((2.0 * params(0))));
	(*out) = A * this->K0cross;
}

void gpmix::CFixedCF::aKcross_diag(VectorXd* out, const CovarInput& Xstar) const throw(CGPMixException)
{
	mfloat_t A = exp((mfloat_t)((2.0 * params(0))));
	(*out) = A*VectorXd::Ones(Xstar.rows());
}


void CFixedCF::aK(MatrixXd *out) const
{
	mfloat_t A = exp((mfloat_t)((2.0 * params(0))));
	(*out) = A * this->K0;
}


void gpmix::CFixedCF::aKgrad_param(MatrixXd *out, const muint_t i) const throw(CGPMixException)
{
	mfloat_t A = exp((mfloat_t)((2.0 * params(0))));

	if (i==0)
	{
		(*out) = 2.0 * A * this->K0;
	}
}

void gpmix::CFixedCF::aKcross_grad_X(MatrixXd *out, const CovarInput & Xstar, const muint_t d) const throw(CGPMixException)
{
	(*out) = MatrixXd::Zero(X.rows(),Xstar.rows());
}

void gpmix::CFixedCF::aKdiag_grad_X(VectorXd *out, const muint_t d) const throw(CGPMixException)
{
	(*out) = VectorXd::Zero(X.rows());
}

MatrixXd CFixedCF::getK0() const
{
	return K0;
}


MatrixXd CFixedCF::getK0cross() const
{
	return K0cross;
}

void CFixedCF::setK0(const MatrixXd& K0)
{
	this->K0 = K0;
}

void CFixedCF::setK0cross(const MatrixXd& Kcross)
{
	this->K0cross = Kcross;
}



void gpmix::CFixedCF::agetK0(MatrixXd *out) const
{
	(*out) = K0;
}



void gpmix::CFixedCF::agetK0cross(MatrixXd *out) const
{
	(*out) = K0cross;
}



void CEyeCF::aKcross(MatrixXd* out, const CovarInput& Xstar ) const throw(CGPMixException)
		{
	(*out).setConstant(Xstar.rows(),X.rows(),0);
		}
void CEyeCF::aKcross_diag(VectorXd* out, const CovarInput& Xstar) const throw(CGPMixException)
		{
	(*out).setConstant(Xstar.rows(),0);
		}
void CEyeCF::aKgrad_param(MatrixXd* out,const muint_t i) const throw(CGPMixException)
		{
	aK(out);
	(*out).diagonal().array()*=2;
		}
void CEyeCF::aKcross_grad_X(MatrixXd* out,const CovarInput& Xstar, const muint_t d) const throw(CGPMixException)
		{
	(*out) = MatrixXd::Zero(X.rows(),Xstar.rows());
}

void CEyeCF::aKdiag_grad_X(VectorXd *out, const muint_t d) const throw(CGPMixException)
{
	(*out) = VectorXd::Zero(X.rows());
}

//other overloads
void CEyeCF::aK(MatrixXd* out) const
{
	mfloat_t A = exp((mfloat_t)((2.0 * params(0))));
	(*out).setConstant(X.rows(),X.rows(),0.0);
	(*out).diagonal().setConstant(A);
}



/* namespace gpmix */
}
