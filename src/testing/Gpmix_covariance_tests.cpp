//============================================================================
// Name        : GPmix.cpp
// Author      :
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include "gpmix/gp/gp_base.h"
#include "gpmix/types.h"
#include "gpmix/matrix/matrix_helper.h"
#include "gpmix/likelihood/likelihood.h"
#include "gpmix/covar/linear.h"
#include "gpmix/gp/gp_base.h"

using namespace std;
using namespace gpmix;
#ifndef PI
#define PI 3.14159265358979323846
#endif



int main() {


	try {
		CCovLinearARD covar(3);
		bool grad_covar = ACovarianceFunction::check_covariance_Kgrad_theta(covar);
		bool grad_x = ACovarianceFunction::check_covariance_Kgrad_x(covar);

		std::cout << grad_covar;
		std::cout << grad_x;

	}
	catch(CGPMixException& e) {
		 cout << e.what() << endl;
	}


}
/*
 * Gpmix_covariance_tests.cpp
 *
 *  Created on: Nov 14, 2011
 *      Author: stegle
 */




