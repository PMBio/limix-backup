/*
 * likelihood.cpp
 *
 *  Created on: Nov 11, 2011
 *      Author: clippert
 */

#include "likelihood.h"
#include "math.h"


namespace limix {

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
    
void CLikNormalNULL::aKhess_param(MatrixXd* out, const muint_t i, const muint_t j) const throw(CGPMixException)
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
	(*out) = MatrixXd::Zero(numRows,numRows);
	assert(getParams().rows() * getParams().cols() > 0);
	(*out).diagonal().setConstant(std::pow((mfloat_t)this->getParams()(0),2));//WARNING: mfloat_t conversion
}

void CLikNormalIso::aKdiag(VectorXd* out) const throw (CGPMixException)
{
	(*out).resize(numRows);
	(*out).setConstant(std::pow(this->getParams()(0),2));
}

void CLikNormalIso::aKcross_diag(VectorXd* out, const CovarInput& Xstar) const throw(CGPMixException)
{
    (*out).resize(Xstar.rows());
    (*out).setConstant(std::pow(this->getParams()(0),2));
}


void CLikNormalIso::aKgrad_param(MatrixXd* out, const muint_t row) const throw (CGPMixException)
{
	(*out).setConstant(numRows,numRows,0.0);
    //(*out) = MatrixXd::Identity(numRows,numRows);
	(*out).diagonal().setConstant(2*this->getParams()(0));
}
    
void CLikNormalIso::aKhess_param(MatrixXd* out, const muint_t i, const muint_t j) const throw(CGPMixException)
{
    if (i>=(muint_t)this->numberParams || j>=(muint_t)this->numberParams)   {
        throw CGPMixException("Parameter index out of range.");
    }
    (*out).setConstant(numRows,numRows,0.0);
    (*out).diagonal().setConstant(2.);
}

/*CLikNormalSVD*/
CLikNormalSVD::CLikNormalSVD() : ALikelihood(2)
{
}

CLikNormalSVD::~CLikNormalSVD()
{
}

void CLikNormalSVD::setX(const CovarInput& X) throw (CGPMixException)
{
	this->numRows = X.rows();
}


void CLikNormalSVD::aK(MatrixXd* out) const throw (CGPMixException)
{
	throw CGPMixException("CLinkNormalSVD cannot be evaluated directly");
}

void CLikNormalSVD::aKdiag(VectorXd* out) const throw (CGPMixException)
{
	throw CGPMixException("CLinkNormalSVD cannot be evaluated directly");
}

void CLikNormalSVD::aKcross_diag(VectorXd* out, const CovarInput& Xstar) const throw(CGPMixException)
{
	throw CGPMixException("CLinkNormalSVD cannot be evaluated directly");
}


void CLikNormalSVD::aKgrad_param(MatrixXd* out, const muint_t row) const throw (CGPMixException)
{
	throw CGPMixException("CLinkNormalSVD cannot be evaluated directly");
}
    
void CLikNormalSVD::aKhess_param(MatrixXd* out, const muint_t i, const muint_t j) const throw(CGPMixException)
{
	throw CGPMixException("CLinkNormalSVD cannot be evaluated directly");
}

mfloat_t CLikNormalSVD::getSigmaK2()
{
	return limix::exp( (mfloat_t)(2.0*this->getParams()(0)));
}


mfloat_t CLikNormalSVD::getDelta()
{
	return limix::exp( (mfloat_t)(2.0*this->getParams()(1)));
}


mfloat_t CLikNormalSVD::getSigmaK2grad()
{
	return 2.0*limix::exp( (mfloat_t)(2.0*this->getParams()(0)));
}
mfloat_t CLikNormalSVD::getDeltagrad()
{
	return 2.0*limix::exp( (mfloat_t)(2.0*this->getParams()(1)));
}




} // end:: namespace limix

