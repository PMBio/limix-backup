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
#include "gpmix/mean/CLinearMean.h"
#include "gpmix/mean/CData.h"

using namespace std;
using namespace gpmix;
#ifndef PI
#define PI 3.14159265358979323846
#endif

#define linMean

int main() {


	try {
		//random input X
		muint_t dim=1;

		MatrixXd X = randn((muint_t)100,(muint_t)dim);
		//y ~ w*X
		MatrixXd w = randn((muint_t)dim,(muint_t)1);
		MatrixXd y = X*w + 0.1*randn((muint_t)100,(muint_t)1);

		CGPHyperParams params;
#ifndef linMean	//dummy mean fucntion
		CData data = CData();
		params["dataTerm"] = MatrixXd();
#else
		params["dataTerm"] = w;
		//Linear Mean Function
		MatrixXd fixedEffects = MatrixXd::Ones((muint_t)100,(muint_t)dim);
		y = fixedEffects*w + y;
		CLinearMean data = CLinearMean(y,w,fixedEffects);
		//data.setParams(w);
		//data.setfixedEffects(fixedEffects);

#endif
		//Ard covariance
		CCovLinearARD covar(X.cols());

		//standard Gaussian lik
		CLikNormalIso lik;

		//GP object
		CGPbase gp(data, covar, lik);
		gp.setY(y);
		gp.setX(X);
		//hyperparams
		CovarInput covar_params = randn(covar.getNumberParams(),(muint_t)1);
		CovarInput lik_params = randn(lik.getNumberParams(),(muint_t)1);

		params["covar"] = covar_params;
		params["lik"] = lik_params;
		//params["X"] = X;


		//get lml and grad
		mfloat_t lml = gp.LML(params);
		CGPHyperParams grad = gp.LMLgrad();

		std::cout <<"lml : "<< lml << "\n";
		std::cout <<"grad[covar] :"<< grad["covar"] << "\n";
#if defined linMean
		std::cout <<"grad[dataTerm] :"<< grad["dataTerm"] << "\n";
#endif
		std::cout <<"grad[lik] :"<< grad["lik"] << "\n";

		CGPopt opt(gp);
		std::cout << "gradcheck: "<< opt.gradCheck() << "\n";
#if 1
		//optimize:
		//construct constraints
		CGPHyperParams upper;
		CGPHyperParams lower;
		upper["lik"] = 5.0*MatrixXd::Ones(1,1);
		lower["lik"] = -5.0*MatrixXd::Ones(1,1);
		opt.setOptBoundLower(lower);
		opt.setOptBoundUpper(upper);
		opt.opt();
#endif






	}
	catch(CGPMixException& e) {
		cout << e.what() << endl;
	}


}



#endif

