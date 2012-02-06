/*
 * likelihood.cpp
 *
 *  Created on: Nov 11, 2011
 *      Author: clippert
 */

#include "likelihood.h"
#include "math.h"


namespace gpmix {

ALikelihood::ALikelihood(muint_t numberParams)
{
	this->numberParams=numberParams;
}

ALikelihood::~ALikelihood() {
	// TODO Auto-generated destructor stub
}

void ALikelihood::aKcross(MatrixXd* out, const CovarInput& Xstar ) const throw(CGPMixException)
{
	(*out) = MatrixXd::Zero(Xstar.rows(),X.rows());
}

void ALikelihood::aKcross_diag(VectorXd* out, const CovarInput& Xstar) const throw(CGPMixException)
		{
		(*out) = VectorXd::Zero(Xstar.rows());
		}

void ALikelihood::aKcross_grad_X(MatrixXd* out,const CovarInput& Xstar, const muint_t d) const throw(CGPMixException)
{
	(*out) = MatrixXd::Zero(Xstar.rows(),X.rows());
}
void ALikelihood::aKdiag_grad_X(VectorXd* out,const muint_t d) const throw(CGPMixException)
{
	(*out) = VectorXd::Zero(X.rows());
}



/*CLikNormalNULL*/
CLikNormalNULL::CLikNormalNULL() : ALikelihood(0)
{
}

CLikNormalNULL::~CLikNormalNULL()
{
}

void CLikNormalNULL::setX(const CovarInput& X) throw (CGPMixException)
{
	this->numRows = X.rows();
}


void CLikNormalNULL::aK(MatrixXd* out) const throw (CGPMixException)
{
	(*out) = MatrixXd::Zero(numRows,numRows);
}

void CLikNormalNULL::aKdiag(VectorXd* out) const throw (CGPMixException)
{
	(*out).resize(numRows);
	(*out).setConstant(0);
}

void CLikNormalNULL::aKcross_diag(VectorXd* out, const CovarInput& Xstar) const throw(CGPMixException)
{
		(*out).setConstant(Xstar.rows(),0.0);
}


void CLikNormalNULL::aKgrad_param(MatrixXd* out, const muint_t row) const throw (CGPMixException)
{
	throw CGPMixException("CLikNormalNULL has no hyperparameters!");
}



/*CLikNormalIso*/

CLikNormalIso::CLikNormalIso() : ALikelihood(1)
{
}

CLikNormalIso::~CLikNormalIso()
{
}

void CLikNormalIso::setX(const CovarInput& X) throw (CGPMixException)
{
	this->numRows = X.rows();
}



void CLikNormalIso::aK(MatrixXd* out) const throw (CGPMixException)
{
	mfloat_t sigma_2 = gpmix::exp( (mfloat_t)(2.0*this->getParams()(0)));//WARNING: mfloat_t conversion
	(*out) = MatrixXd::Zero(numRows,numRows);
	(*out).diagonal().setConstant(sigma_2);
}

void CLikNormalIso::aKdiag(VectorXd* out) const throw (CGPMixException)
{
	mfloat_t sigma_2 = gpmix::exp( (mfloat_t)(2.0*this->getParams()(0)));//WARNING: mfloat_t conversion
	(*out).resize(numRows);
	(*out).setConstant(sigma_2);
}

void CLikNormalIso::aKcross_diag(VectorXd* out, const CovarInput& Xstar) const throw(CGPMixException)
{
		mfloat_t sigma_2 = gpmix::exp( (mfloat_t)(2.0*this->getParams()(0)));//WARNING: mfloat_t conversion
		(*out).setConstant(Xstar.rows(),sigma_2);
}


void CLikNormalIso::aKgrad_param(MatrixXd* out, const muint_t row) const throw (CGPMixException)
{
	mfloat_t sigma_2 = 2.0*gpmix::exp( (mfloat_t)(2.0*this->getParams()(0)));//WARNING: mfloat_t conversion

	(*out) = MatrixXd::Zero(numRows,numRows);
	(*out).diagonal().setConstant(sigma_2);
}

} // end:: namespace gpmix

