/*
 * gp_base.cpp
 *
 *  Created on: Nov 11, 2011
 *      Author: stegle
 */

#include "gp_base.h"

namespace gpmix {

	CGPbase::CGPbase(ACovarianceFunction& covar) : covar(covar) {
		// TODO Auto-generated constructor stub
	}

	CGPbase::~CGPbase() {
		// TODO Auto-generated destructor stub
	}

	double CGPbase::LML(CGPHyperParams& hyperparams){

		CovarInput x1, x2;	//TODO: These are dummies
		CovarParams params = hyperparams.get("covar");

		this->covar.K(params, x1, x2);
		return 0.0;
	}
	CGPHyperParams CGPbase::LMLgrad(CGPHyperParams& hyperparams){
		CGPHyperParams grad;
		return grad;
	}

} /* namespace gpmix */




