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
		MatrixXd X = randn((muint_t)10,(muint_t)3);

		//1. linear covariance ISO
		CCovLinearISO covar1;
		gradcheck(covar1,X);

		//2. ard covariance
		CCovLinearARD covar2;
		gradcheck(covar2,X);

		//3. se covariance
		CCovSqexpARD covar3;
		gradcheck(covar3,X);

		//4. fixed CF
		CFixedCF covar4;
		covar4.setK0(X*X.transpose());
		gradcheck(covar4,X);

		//4. combinators
	}
	catch(CGPMixException& e) {
		cout << e.what() << endl;
	}


}



#endif

