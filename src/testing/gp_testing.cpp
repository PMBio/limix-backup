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



int main() {


	try {
		//random input X
		MatrixXd X = randn((muint_t)10,(muint_t)2);

		//Ard covariance
		CCovLinearARD covar2(X.cols());



	}
	catch(CGPMixException& e) {
		cout << e.what() << endl;
	}


}



#endif

