//============================================================================
// Name        : GPmix.cpp
// Author      :
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include "gpmix/types.h"
#include "gpmix/covar/covariance.h"
#include "gpmix/covar/linear.h"
#include "gpmix/utils/matrix_helper.h"
#include "gpmix/LMM/lmm.h"
  using namespace gpmix;

using namespace std;
using namespace gpmix;

int main() {

	MatrixXd snps = (MatrixXd)randn((muint_t)100,(muint_t)1000);
	MatrixXd pheno = (MatrixXd)randn((muint_t)100,(muint_t)1);
	MatrixXd covs = MatrixXd::Ones(100,1);

	MatrixXd K = 1.0/snps.cols() * (snps*snps.transpose());

	CLmm lmm;
	lmm.setK(K);
	lmm.setSNPs(snps);
	lmm.setPheno(pheno);
	lmm.setCovs(covs);

	lmm.process();

}
