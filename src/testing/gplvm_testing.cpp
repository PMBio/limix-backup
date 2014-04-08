// Copyright(c) 2014, The LIMIX developers (Christoph Lippert, Paolo Francesco Casale, Oliver Stegle)
// All rights reserved.
//
// LIMIX is provided under a 2-clause BSD license.
// See license.txt for the complete license.

#if 0

#include <iostream>
#include "limix/gp/gp_base.h"
#include "limix/gp/gp_opt.h"
#include "limix/types.h"
#include "limix/likelihood/likelihood.h"
#include "limix/gp/gp_base.h"
#include "limix/utils/matrix_helper.h"
#include "limix/covar/linear.h"
#include "limix/covar/se.h"
#include "limix/covar/fixed.h"
#include "limix/covar/combinators.h"


using namespace std;
using namespace limix;
#ifndef PI
#define PI 3.14159265358979323846
#endif



int main() {


	try {
		//random input X
		muint_t K=2;
		muint_t D=10;
		muint_t N=50;
		mfloat_t eps = 0.1;

		MatrixXd X = randn((muint_t)N,(muint_t)K);
		//y ~ w*X
		MatrixXd w = randn((muint_t)K,(muint_t)D);
		MatrixXd y = X*w + eps*randn((muint_t)N,(muint_t)D);

		//Ard covariance
		CCovLinearISO covar(X.cols());
		//standard Gaussian lik
		CLikNormalIso lik;

		//GP object
		CGPbase gp(covar,lik);
		gp.setY(y);
		gp.setX(X);

		//hyperparams
		CovarInput covar_params = randn(covar.getNumberParams(),(muint_t)1);
		CovarInput lik_params = randn(lik.getNumberParams(),(muint_t)1);
		CGPHyperParams params;
		params["covar"] = covar_params;
		params["lik"] = lik_params;
		params["X"] = X;

		//get lml and grad
		mfloat_t lml = gp.LML(params);
		CGPHyperParams grad = gp.LMLgrad();

		std::cout << lml << "\n";
		std::cout << grad["covar"] << "\n";
		std::cout << grad["lik"] << "\n";

		CGPopt opt(gp);
		std::cout << "gradcheck: "<< opt.gradCheck();
#if 0
		//optimize:
		opt.opt();
#endif






	}
	catch(CGPMixException& e) {
		cout << e.what() << endl;
	}


}



#endif

