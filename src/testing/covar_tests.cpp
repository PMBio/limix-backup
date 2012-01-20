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
#include "gpmix/covar/freeform.h"



using namespace std;
using namespace gpmix;
#ifndef PI
#define PI 3.14159265358979323846
#endif


void gradcheck(ACovarianceFunction& covar,CovarInput X)
{
	//create random params:
	if (!isnull(X))
	{
		covar.setX(X);
	}
	CovarInput params = randn(covar.getNumberParams(),(muint_t)1);
	covar.setParams(params);
	bool grad_covar = ACovarianceFunction::check_covariance_Kgrad_theta(covar);
	bool grad_x = true;
	if (!isnull(X))
		ACovarianceFunction::check_covariance_Kgrad_x(covar,1E-5,1E-2,true);
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
		gradcheck(covar4,MatrixXd());


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

		//5. combinators: create product of 2 covariances
		CProductCF covar6;
		CCovLinearISO covar6_1(X.cols());
		CCovSqexpARD  covar6_2(X.cols());
		covar6.addCovariance(&covar6_1);
		covar6.addCovariance(&covar6_2);
		//create combinatin of X
		MatrixXd X3 = MatrixXd::Zero(X.rows(),2*X.cols());
		X3.block(0,0,X.rows(),X.cols()) = X;
		X3.block(0,X.cols(),X.rows(),X.cols()) = X;
		//setX
		covar6.setX(X3);
		//draw random params
		CovarParams params2 = randn(covar6.getNumberParams(),(muint_t)1);
		MatrixXd test2 = covar6.getX();
		covar6.setParams(params2);
		gradcheck(covar6,X2);


		//7. Freeform
		CCovFreeform covar7(3);
		MatrixXd X7 = MatrixXd::Zero(6,1);
		X7(0,0)=0;
		X7(1,0)=0;
		X7(1,0)=1;
		X7(2,0)=1;
		X7(3,0)=2;
		X7(4,0)=2;

		covar7.setX(X7);

		CovarParams params7 = VectorXd::LinSpaced(covar7.getNumberParams(),1,7);
		covar7.setParams(params7);

		MatrixXd K7 = covar7.K();

		std::cout << K7 << "\n";
		//draw random params
		gradcheck(covar7,MatrixXd());


		//8. Eye
		CEyeCF covar8;
		MatrixXd X8 = MatrixXd::Zero(6,0);
		covar8.setX(X8);

		CovarParams params8 = VectorXd::LinSpaced(covar8.getNumberParams(),1,7);
		covar8.setParams(params8);

		MatrixXd K8 = covar8.K();

		std::cout << K8 << "\n";
		//draw random params
		gradcheck(covar8,MatrixXd());


	}
	catch(CGPMixException& e) {
		cout << e.what() << endl;
	}


}



#endif

