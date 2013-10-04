//============================================================================
// Name        : GPmix.cpp
// Author      :
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#if 1

#include <iostream>
#include "limix/gp/gp_base.h"
#include "limix/types.h"
#include "limix/likelihood/likelihood.h"
#include "limix/gp/gp_base.h"
#include "limix/utils/matrix_helper.h"
#include "limix/covar/linear.h"
#include "limix/covar/se.h"
#include "limix/covar/combinators.h"
#include "limix/covar/freeform.h"

#include "limix/utils/cache.h"



using namespace std;
using namespace limix;
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


	MatrixXd K1 = randn(10,10);
	MatrixXd K2 = randn(20,20);
	PFixedCF f1 = PFixedCF(new CFixedCF(K1));
	PFixedCF f2 = PFixedCF(new CFixedCF(K2));

	PKroneckerCF C = PKroneckerCF(new CKroneckerCF());
	C->setRowCovariance(f1);
	C->setColCovariance(f2);

	std::cout << f1->K();

	MatrixXd K = C->K();

	std::cout << K.rows() << "," << K.cols() << "\n";

}



#endif

