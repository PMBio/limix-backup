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
#include "gpmix/LMM/lmm_old.h"
  using namespace gpmix;

using namespace std;
using namespace gpmix;

int main() {

	int n = 100;
	int p = 10;
	int s = 200;
	MatrixXd snps = (MatrixXd)randn((muint_t)n,(muint_t)s);
	MatrixXd pheno = (MatrixXd)randn((muint_t)n,(muint_t)p);
	MatrixXd covs = MatrixXd::Ones(n,1);

	MatrixXd K = 1.0/snps.cols() * (snps*snps.transpose());

	//Default settings:
	int num_intervals0 = 100;
	int num_intervalsAlt = 0;
	double ldeltamin0 = -5;
	double ldeltamax0 = 5;
	double ldeltaminAlt = -1.0;
	double ldeltamaxAlt =1.0;
	MatrixXd pvals = MatrixXd(p, s);

	if (0){
		lmm_old::train_associations(&pvals, snps, pheno,	K, covs, num_intervalsAlt,ldeltaminAlt, ldeltamaxAlt, num_intervals0, ldeltamin0, ldeltamax0);
		cout << "pv_old:\n"<<scientific<<pvals<<endl;
	}

	if (0){
		CLmm lmm;

		lmm.setK(K);
		lmm.setSNPs(snps);
		lmm.setPheno(pheno);
		lmm.setCovs(covs);

		lmm.process();
		MatrixXd pv = lmm.getPv();
		cout <<"pv_new:\n"<< scientific <<pv<<endl;
	}

	if(1)
	{
		MatrixXd Kp = 1.0/p * (snps.block(0,0,p,s)*snps.block(0,0,p,s).transpose());
		//cout <<Kp;
		CKroneckerLMM kron;
		kron.setK_R(K);
		kron.setK_C(Kp);
		kron.setPheno(pheno);
		kron.setSNPs(snps);
		kron.setCovs(covs);

		kron.process();

		MatrixXd pv = kron.getPv().block(0,0,1,s);
		cout <<"pv_kron:\n"<< scientific <<pv<<endl;
	}
	cout << "finished";

}
