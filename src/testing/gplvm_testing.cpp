//============================================================================
// Name        : GPmix.cpp
// Author      :
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#if 0

#include <iostream>
#include "gpmix/gp/gp_base.h"
#include "gpmix/gp/gp_opt.h"
#include "gpmix/types.h"
#include "gpmix/likelihood/likelihood.h"
#include "gpmix/gp/gp_base.h"
#include "gpmix/utils/matrix_helper.h"
#include "gpmix/covar/linear.h"
#include "gpmix/covar/se.h"
#include "gpmix/covar/fixed.h"
#include "gpmix/covar/combinators.h"


using namespace std;
using namespace gpmix;
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

