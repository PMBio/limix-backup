//============================================================================
// Name        : GPmix.cpp
// Author      :
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#if 1

#include <iostream>
#include "gpmix/gp/gp_base.h"
#include "gpmix/gp/gp_opt.h"
#include "gpmix/types.h"
#include "gpmix/likelihood/likelihood.h"
#include "gpmix/gp/gp_base.h"
#include "gpmix/gp/gp_kronecker.h"
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
		muint_t Kr=2;
		muint_t Kc=3;

		muint_t D=10;
		muint_t N=50;
		mfloat_t eps = 0.1;

		MatrixXd X = randn((muint_t)N,(muint_t)Kr);
		//y ~ w*X
		MatrixXd w = randn((muint_t)Kr,(muint_t)D);
		MatrixXd y = X*w + eps*randn((muint_t)N,(muint_t)D);


		MatrixXd Xr = X;
		MatrixXd Xc = randn((muint_t)D,Kc);


		//covariances
		CCovLinearISO covar_r(Kr);
		CCovLinearISO covar_c(Kc);
		//likelihood
		CLikNormalIso lik;

		//GP object
		CGPkronecker gp(covar_r,covar_c,lik);
		gp.setY(y);
		gp.setX_r(Xr);
		gp.setX_c(Xc);

		//hyperparams
		CovarInput covar_params_r = randn(covar_r.getNumberParams(),(muint_t)1);
		CovarInput covar_params_c = randn(covar_c.getNumberParams(),(muint_t)1);
		CovarInput lik_params = randn(lik.getNumberParams(),(muint_t)1);
		CGPHyperParams params;
		params["covar_r"] = covar_params_r;
		params["covar_c"] = covar_params_c;
		params["lik"] = lik_params;
#if 0
		params["X_r"] = Xr;
		params["X_c"] = Xc;
#endif

		//get lml and grad
		mfloat_t lml = gp.LML(params);
		CGPHyperParams grad = gp.LMLgrad();

		std::cout << lml << "\n";
		std::cout << grad["covar_r"] << "\n";
		std::cout << grad["covar_c"] << "\n";
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

