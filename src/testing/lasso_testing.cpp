// Copyright(c) 2014, The LIMIX developers (Christoph Lippert, Paolo Francesco Casale, Oliver Stegle)
// All rights reserved.
//
// LIMIX is provided under a 2-clause BSD license.
// See license.txt for the complete license.

#if 0

#include <iostream>
#include "limix/gp/gp_base.h"
#include "limix/types.h"
#include "limix/utils/matrix_helper.h"
#include "limix/likelihood/likelihood.h"
#include "limix/covar/linear.h"
#include "limix/gp/gp_base.h"
#include "limix/lasso/lasso.h"


using namespace std;
using namespace limix;

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
