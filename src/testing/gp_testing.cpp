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
#include "gpmix/mean/CKroneckerMean.h"
#include "gpmix/mean/CSumLinear.h"
#include "gpmix/mean/CData.h"

using namespace std;
using namespace gpmix;
#ifndef PI
#define PI 3.14159265358979323846
#endif

#define linMean
#define kronMean
#define sumKronMean

int main() {


	try {
		//random input X
		muint_t dim=1;
		muint_t nsamples =100;
		muint_t targets = 2;
		muint_t nWeights = 1;

		MatrixXd X = randn(nsamples,dim);
		//y ~ w*X
		MatrixXd w = randn(dim,nWeights);
		MatrixXd A = randn(nWeights, targets);
		MatrixXd y = X*w*A + 0.1*randn(nsamples,targets);

		CGPHyperParams params;
#ifndef linMean	//dummy mean fucntion
		CData data = CData();
		MatrixXd w_ = MatrixXd();
#else
		//Linear Mean Function
		MatrixXd fixedEffects = MatrixXd::Ones((muint_t)nsamples,(muint_t)dim);
		y = fixedEffects*w*A + y;
		MatrixXd w_ = w*A;
		CLinearMean data_1 = CLinearMean(y,w_,fixedEffects);

		//Kronecker Mean Function
		MatrixXd w__ = w;
		CKroneckerMean data_2 = CKroneckerMean(y,w__,fixedEffects,A);

		CSumLinear dataSum = CSumLinear();
		dataSum.appendTerm(data_1);
		dataSum.appendTerm(data_2);
#endif
		//Ard covariance
		CCovLinearARD covar(X.cols());

		//standard Gaussian lik
		CLikNormalIso lik;

		//GP object
		CGPbase gp(dataSum, covar, lik);
		gp.setY(y);
		gp.setX(X);
		//hyperparams
		CovarInput covar_params = randn(covar.getNumberParams(),(muint_t)1);
		CovarInput lik_params = randn(lik.getNumberParams(),(muint_t)1);

		params["covar"] = covar_params;
		params["lik"] = lik_params;
		//params["X"] = X;
		//(w_.rows() * w_.cols()) +
		params["dataTerm"] = MatrixXd::Zero( (w_.rows() * w_.cols()) + (w__.rows() * w__.cols()), 1);

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


		gp.predictMean(X.block(0,0,10,X.cols()));
		gp.predictVar(X.block(0,0,10,X.cols()));



	}
	catch(CGPMixException& e) {
		cout << e.what() << endl;
	}


}



#endif

