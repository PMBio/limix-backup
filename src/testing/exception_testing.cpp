//============================================================================
// Name        : GPmix.cpp
// Author      :
// Version     :
// Copyright   : Your copyright notice
// Test file for kronecker LMM test
//============================================================================

#if 0

//#define debugkron 1

#include <iostream>
#include "limix/types.h"
#include "limix/LMM/kronecker_lmm.h"

using namespace std;
using namespace limix;
#ifndef PI
#define PI 3.14159265358979323846
#endif

#define GPLVM

int main() {

	try{
		CLMM lmm;
		int n = 100;
		int p = 1;
		int s = 10;
		int ncov = 1;

		MatrixXd snps = (MatrixXd)randn((muint_t)n,(muint_t)s);
		MatrixXd pheno = (MatrixXd)randn((muint_t)(n+1),(muint_t)p);
		MatrixXd covs = MatrixXd::Ones(n,ncov);

		MatrixXd K = 1.0/snps.cols() * (snps*snps.transpose());

		VectorXd v = VectorXd::Ones(3);
		MatrixXd M = MatrixXd::Ones(3,2);

		lmm.setK(K);
		lmm.setSNPs(snps);
		lmm.setPheno(pheno);
		lmm.setCovs(covs);
		lmm.process();


		lmm.setSNPs(snps);
		lmm.process();
		
		}
		catch(CLimixException& e) {
			cout <<"Caught: Exception : "<< e.what() << endl;
		}

}

#endif
