/*
 * CGPlvm.cpp
 *
 *  Created on: Jan 2, 2012
 *      Author: stegle
 */

#include "gp_lvm.h"
#include "gpmix/utils/matrix_helper.h"


namespace gpmix {

CGPlvm::CGPlvm(ACovarianceFunction& covar, ALikelihood& lik) : CGPbase(covar,lik)
{
	// TODO Auto-generated constructor stub

}

CGPlvm::~CGPlvm() {
	// TODO Auto-generated destructor stub
}

void CGPlvm::setX(const CovarInput& X) throw (CGPMixException)
{
	CGPbase::setX(X);
	//determine default GPLVM dimensions if appropriate:
	if (isnull(gplvmDimensions))
		this->gplvmDimensions = VectorXi::LinSpaced(X.rows(),0,X.rows());
}


void CGPlvm::updateParams() throw (CGPMixException)
{
	//1. update covar and lik
	CGPbase::updateParams();
	//2. update X if available
	if (params.exists("X"))
	{
		//check dimensions match
		MatrixXd& X = params["X"];
		if (X.cols()!=gplvmDimensions.cols())
		{
			ostringstream os;
			os << "CGPLvm X param update dimension missmatch. X("<<X.rows()<<","<<X.cols()<<") <-> gplvm_dimensions:"<<gplvmDimensions.cols()<<"!";
			throw CGPMixException(os.str());
		}
		//update
		for (muint_t ic=0;ic<(muint_t)X.cols();ic++)
			this->covar.setXcol(X.col(ic),gplvmDimensions(ic));
	}
}

CGPHyperParams CGPlvm::LMLgrad() throw (CGPMixException)
{
	//gradient for lik and covar:
	CGPHyperParams rv = CGPbase::LMLgrad();
	VectorXd grad_X;
	aLMLgrad_X(&grad_X);
	rv.set("X",grad_X);
	return rv;
}


void CGPlvm::aLMLgrad_X(VectorXd* out) throw (CGPMixException)
{
	//0. set output dimensions
	(*out).resize(this->getNumberSamples(),this->gplvmDimensions.cols());

	//1. get W:
	MatrixXd* W = this->getDKinv_KinvYYKinv();
	//loop through GLVM dimensions and calculate gradient

	MatrixXd WKgrad_X;
	VectorXd Kdiag_grad_X;
	for (muint_t ic=0;ic<(muint_t)this->gplvmDimensions.cols();ic++)
	{
		muint_t col = gplvmDimensions(ic);
		//get gradient
		covar.aKgrad_X(&WKgrad_X,col);
		covar.aKdiag_grad_X(&Kdiag_grad_X,col);
		WKgrad_X.diagonal() = Kdiag_grad_X;
		//precalc elementwise product of W and K
		WKgrad_X.array()*=(*W).array();
		(*out).col(ic) = 0.5* (2*WKgrad_X.colwise().sum() - WKgrad_X.diagonal());
	}
}



} /* namespace gpmix */
