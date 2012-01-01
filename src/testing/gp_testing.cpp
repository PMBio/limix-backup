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
		muint_t dim=1;

		MatrixXd X = randn((muint_t)100,(muint_t)dim);
		//y ~ w*X
		MatrixXd w = randn((muint_t)dim,(muint_t)1);
		MatrixXd y = X*w + 0.1*randn((muint_t)100,(muint_t)1);

		//Ard covariance
		CCovLinearARD covar(X.cols());

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
		params.set("covar",covar_params);
		params.set("lik",lik_params);

		//get lml and grad
		mfloat_t lml = gp.LML(params);
		CGPHyperParams grad = gp.LMLgrad();

		std::cout << lml << "\n";
		std::cout << grad["covar"] << "\n";
		std::cout << grad["lik"] << "\n";

		//optimize:
		CGPopt opt(gp);
		opt.opt();







	}
	catch(CGPMixException& e) {
		cout << e.what() << endl;
	}


}



#endif

