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


void gradcheck(ACovarianceFunction& covar,CovarInput X)
{
	//create random params:
	covar.setX(X);
	CovarInput params = randn(covar.getNumberParams(),(muint_t)1);
	covar.setParams(params);
	bool grad_covar = ACovarianceFunction::check_covariance_Kgrad_theta(covar);
	bool grad_x = ACovarianceFunction::check_covariance_Kgrad_x(covar,1E-5,1E-2,true);
	std::cout << "GradCheck: " << covar.getName();
	std::cout << grad_covar;
	std::cout << grad_x << "\n";
}



int main() {


	try {
		//random input X
		MatrixXd X = randn((muint_t)3,(muint_t)4);

		//0. Gauss Lk
		CLikNormalIso lik1;
		gradcheck(lik1,X);


		//1. linear covariance ISO
		CCovLinearISO covar1(X.cols());
		gradcheck(covar1,X);

		//2. ard covariance
		CCovLinearARD covar2(X.cols());
		gradcheck(covar2,X);

		//3. se covariance
		CCovSqexpARD covar3(X.cols());
		gradcheck(covar3,X);

		//4. fixed CF
		CFixedCF covar4;
		covar4.setK0(X*X.transpose());
		gradcheck(covar4,MatrixXd::Zero(0,0));


		//4. combinators: create sum of 2 covariances
		CSumCF covar5;
		CCovLinearISO covar5_1(X.cols());
		CCovSqexpARD  covar5_2(X.cols());
		covar5.addCovariance(&covar5_1);
		covar5.addCovariance(&covar5_2);
		//create combinatin of X
		MatrixXd X2 = MatrixXd::Zero(X.rows(),2*X.cols());
		X2.block(0,0,X.rows(),X.cols()) = X;
		X2.block(0,X.cols(),X.rows(),X.cols()) = X;
		//setX
		covar5.setX(X2);
		//draw random params
		CovarParams params = randn(covar5.getNumberParams(),(muint_t)1);
		MatrixXd test = covar5.getX();
		covar5.setParams(params);
		gradcheck(covar5,X2);

	}
	catch(CGPMixException& e) {
		cout << e.what() << endl;
	}


}



#endif

