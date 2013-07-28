/*
 * fixed.cpp
 *
 *  Created on: Nov 11, 2011
 *      Author: stegle
 */
#include "fixed.h"
#include "limix/types.h"
#include "limix/utils/matrix_helper.h"
#include "assert.h"

namespace limix {



CFixedCF::CFixedCF(const MatrixXd & K0) : ACovarianceFunction(1)
{
	this->K0 = K0;
}


CFixedCF::~CFixedCF()
{
}

muint_t CFixedCF::Kdim() const throw(CGPMixException)
{
	if(isnull(K0))
	{
		throw CGPMixException("FixedCF: Kdim cannot be evaluated before K0 is defined");
	}
	return K0.rows();
}


void CFixedCF::aKcross(MatrixXd *out, const CovarInput & Xstar) const throw(CGPMixException)
{
	(*out) = params(0) * this->K0cross;
}

void CFixedCF::aKcross_diag(VectorXd* out, const CovarInput& Xstar) const throw(CGPMixException)
{
	(*out) = params(0)*K0cross_diag;
}


void CFixedCF::aK(MatrixXd *out) const throw (CGPMixException)
{
	(*out) = params(0) * this->K0;
}


void CFixedCF::aKgrad_param(MatrixXd *out, const muint_t i) const throw(CGPMixException)
{
	mfloat_t Agrad = 1;
	if (i==0)
	{
		(*out) = Agrad*this->K0;
	}
}
    
void CFixedCF::aKhess_param(MatrixXd* out, const muint_t i, const muint_t j) const throw(CGPMixException)
{
    if (i>=(muint_t)this->numberParams || j>=(muint_t)this->numberParams)   {
        throw CGPMixException("Parameter index out of range.");
    }
    (*out)=MatrixXd::Zero(this->K0.rows(),this->K0.rows());
}

void CFixedCF::aKcross_grad_X(MatrixXd *out, const CovarInput & Xstar, const muint_t d) const throw(CGPMixException)
{
	(*out) = MatrixXd::Zero(X.rows(),Xstar.rows());
}

void CFixedCF::aKdiag_grad_X(VectorXd *out, const muint_t d) const throw(CGPMixException)
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



void CFixedCF::agetK0(MatrixXd *out) const
{
	(*out) = K0;
}



void CFixedCF::agetK0cross(MatrixXd *out) const
{
	(*out) = K0cross;
}

void CFixedCF::agetK0cross_diag(VectorXd *out) const
{
	(*out) = K0cross_diag;
}

void CFixedCF::agetParamBounds0(CovarParams* lower, CovarParams* upper) const
{
	//bounding: [0,inf]
	*lower = VectorXd::Ones(getNumberParams())*0;
	*upper = VectorXd::Ones(getNumberParams())*INFINITY;
}

VectorXd CFixedCF::getK0cross_diag() const
{
	 return K0cross_diag;
}

void CFixedCF::setK0cross_diag(const VectorXd& Kcross_diag)
{
	this->K0cross_diag = Kcross_diag;
}



void CEyeCF::aKcross(MatrixXd* out, const CovarInput& Xstar ) const throw(CGPMixException)
{
	(*out).setConstant(Xstar.rows(),this->EyeDimension,0);
}
void CEyeCF::aKcross_diag(VectorXd* out, const CovarInput& Xstar) const throw(CGPMixException)
{
	(*out).setConstant(Xstar.rows(),0);
}
void CEyeCF::aKgrad_param(MatrixXd* out,const muint_t i) const throw(CGPMixException)
{
	(*out)=MatrixXd::Identity(this->EyeDimension,this->EyeDimension);
}
void CEyeCF::aKhess_param(MatrixXd* out, const muint_t i, const muint_t j) const throw(CGPMixException)
{
    if (i>=(muint_t)this->numberParams || j>=(muint_t)this->numberParams)   {
        throw CGPMixException("Parameter index out of range.");
    }
    (*out)=MatrixXd::Zero(this->EyeDimension,this->EyeDimension);
}
void CEyeCF::aKcross_grad_X(MatrixXd* out,const CovarInput& Xstar, const muint_t d) const throw(CGPMixException)
{
	(*out) = MatrixXd::Zero(Xstar.rows(),this->EyeDimension);
}

void CEyeCF::aKdiag_grad_X(VectorXd *out, const muint_t d) const throw(CGPMixException)
{
	(*out) = VectorXd::Zero(this->EyeDimension);
}

//other overloads
void CEyeCF::aK(MatrixXd* out) const throw (CGPMixException)
{
	(*out).setConstant(this->EyeDimension,this->EyeDimension,0.0);
	(*out).diagonal().setConstant(params(0));
}
    
    
void CEyeCF::setEyeDimension(const muint_t EyeDim)
{
    this->EyeDimension=EyeDim;
}

muint_t CEyeCF::getEyeDimension()
{
    return this->EyeDimension;
}
    
    

/* namespace limix */
}
