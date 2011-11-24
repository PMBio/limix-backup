//============================================================================
// Name        : GPmix.cpp
// Author      :
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#define SWIG_FILE_WITH_INIT
#define SWIG
#include "gpmix/types.h"
#include "gpmix/covar/covariance.h"
#include "gpmix/covar/linear.h"
#include "gpmix/LMM/lmm.h"
  using namespace gpmix;

using namespace std;
using namespace gpmix;

int main() {

	//CCovLinearISO cov();

	//gpmix::CCovLinearISO* cov = new gpmix::CCovLinearISO();
	std::cout << "hi";


	/*
	MatrixXd X = randn((muint_t) 10,(muint_t)3);
	CCovLinearISO covar(3);

	VectorXd hyperparams = VectorXd (1);
	hyperparams(0) = 2.0;

	std::cout << X;

	MatrixXd K = covar.Kgrad_theta(hyperparams,X,0);
	std::cout << K;
	*/
}
