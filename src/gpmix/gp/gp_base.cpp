/*
 * gp_base.cpp
 *
 *  Created on: Nov 11, 2011
 *      Author: stegle
 */

#include "gp_base.h"

#ifndef PI
#define PI 3.14159265358979323846
#endif

namespace gpmix {

	CGPbase::CGPbase(ACovarianceFunction& covar, ALikelihood& lik) : covar(covar), lik(lik) {
		// TODO Auto-generated constructor stub
	}

	CGPbase::~CGPbase() {
		// TODO Auto-generated destructor stub
	}

	float_t CGPbase::LML(CGPHyperParams& hyperparams){

		//TODO: get X
		CovarInput X;	//TODO: These are dummies
		MatrixXd K = this->covar.K(hyperparams.get("covar"), X, X);
		this->lik.applyToK(hyperparams.get("lik"), K);
		LDLT<MatrixXd> chol(K);

		//TODO get Y
		MatrixXd Y;
		MatrixXd KinvY = chol.solve(Y);

		int_t nX = X.rows();
		int_t nY = Y.rows();

		float_t lml_quad = 0.0;
		float_t lml_det = 0.0;

		float_t lml_const = 0.5 * nY * nX * std::log((2.0 * PI));

		//= K.ldlt();
		return lml_quad + lml_det + lml_const;
	}
	CGPHyperParams CGPbase::LMLgrad(CGPHyperParams& hyperparams){
		CGPHyperParams grad;
		return grad;
	}

} /* namespace gpmix */
