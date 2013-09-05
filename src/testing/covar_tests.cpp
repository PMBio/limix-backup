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


	try {

		//test the cache
		PNamedCache cache (new CNamedCache());
		MatrixXd m =randn(10,10);


		PMatrixXd pm = PMatrixXd(new MatrixXd(m));
		cache->set("m",pm);
		PCVoid value = cache->get("m");
		//PMatrixXd pm2 = static_pointer_cast<const MatrixXd>(value);

		PCVoid test = pm;

		PConstMatrixXd pm3 = static_pointer_cast<const MatrixXd>(value);

		std::cout << *pm3;

	}
	catch(CGPMixException& e) {
		cout << e.what() << endl;
	}


}



#endif

