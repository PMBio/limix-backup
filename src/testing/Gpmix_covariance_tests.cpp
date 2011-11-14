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
		//1. create input for linear covaraince
		MatrixXd X = randn((uint_t)10,(uint_t)3);
		//2. create covaraince parmaeteres

		CCovLinearISO covar(3);
		//get random hyperparamters
		CovarParams params = randn(covar.hyperparameters(),1);

		MatrixXd K =  covar.K(params,X,X);
		std::cout<< K;

		check_covariance_Kgrad_theta(covar,params,X);

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




