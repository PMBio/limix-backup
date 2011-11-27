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
#include "gpmix/utils/matrix_helper.h"
  using namespace gpmix;

using namespace std;
using namespace gpmix;

void test(MatrixXd& m)
{
	m(1,1) = 3;
}

int main() {

	//CCovLinearISO cov();

	gpmix::CCovLinearISO cov;

	//set Parmas
	CovarParams p = cov.getParams();
	p(0) = 1.0;
	cov.setParams(p);
	//set inputs
	CovarInput X = (MatrixXd)randn((muint_t)100,(muint_t)3);
	cov.setX(X);

	bool check_grad = ACovarianceFunction::check_covariance_Kgrad_theta(cov);

	std::cout << check_grad;
	//Eigen::Map<MatrixXdscipy>(out_data,10,1) T;

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
