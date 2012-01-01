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
		MatrixXd X = randn((muint_t)10,(muint_t)3);
		MatrixXd y = randn((muint_t)10,(muint_t)1);

		//Ard covariance
		CCovLinearARD covar(X.cols());

		//standard Gaussian lik
		CLikNormalIso lik;

		//GP object
		CGPbase gp(covar,lik);
		gp.setY(y);
		//hyperparams
		CovarInput covar_params = randn(covar.getNumberParams(),(muint_t)1);
		CovarInput lik_params = randn(lik.getNumberParams(),(muint_t)1);
		covar.setX(X);
		lik.setX(X);
		covar.setParams(covar_params);
		lik.setParams(lik_params);
		//
		mfloat_t lml = gp.LML();
		VectorXd grad_covar;
		gp.aLMLgrad_covar(&grad_covar);
		VectorXd grad_lik;
		gp.aLMLgrad_lik(&grad_lik);

		std::cout << lml << "\n";
		std::cout << grad_covar << "\n";
		std::cout << grad_lik << "\n";


		CGPHyperParams params;
		params.set("covar",covar_params);
		params.set("lik",lik_params);

		gp.setParams(params);





	}
	catch(CGPMixException& e) {
		cout << e.what() << endl;
	}


}



#endif

