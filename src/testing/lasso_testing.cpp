

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
#include "gpmix/utils/matrix_helper.h"
#include "gpmix/likelihood/likelihood.h"
#include "gpmix/covar/linear.h"
#include "gpmix/gp/gp_base.h"
#include "gpmix/lasso/lasso.h"


using namespace std;
using namespace gpmix;

int main() {
	cout << "!!!Hello World!!!" << endl; // prints !!!Hello World!!!


	MatrixXd X = gpmix::randn((muint_t)100,(muint_t)100);
	MatrixXd y = gpmix::randn((muint_t)100,(muint_t)1);

	MatrixXd w;
	mfloat_t mu = 10;
	lasso_irr(&w,X,y,mu,1E-4,1E-4,10);

	return 0;
}

#endif
