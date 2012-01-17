
#if 0
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
	int p = 2;
	int s = 5;
	int ncov = 1;
	MatrixXd snps = (MatrixXd)randn((muint_t)n,(muint_t)s);
	MatrixXd pheno = (MatrixXd)randn((muint_t)n,(muint_t)p);
	MatrixXd covs = MatrixXd::Ones(n,ncov);

	MatrixXd K = 1.0/snps.cols() * (snps*snps.transpose());

	//Default settings:
	int num_intervals0 = 100;
	int num_intervalsAlt = 0;
	double ldeltamin0 = -5;
	double ldeltamax0 = 5;
	double ldeltaminAlt = -1.0;
	double ldeltamaxAlt =1.0;
	MatrixXd pvals = MatrixXd(p, s);

	if (1){ //LMM testing using old code
		lmm_old::train_associations(&pvals, snps, pheno,	K, covs, num_intervalsAlt,ldeltaminAlt, ldeltamaxAlt, num_intervals0, ldeltamin0, ldeltamax0);
		cout << "pv_old:\n"<<scientific<<pvals<<endl;
	}



	if (0){ //Simple Kronecker LMM
		MatrixXd Kp = 1.0/p * (snps.block(0,0,p,s)*snps.block(0,0,p,s).transpose());

		MatrixXd Wkron0 = MatrixXd::Ones(p,1);
		MatrixXd Wkron  = MatrixXd::Zero(p,1);
		//select one phenotype for testing only:
		Wkron(0,0) = 1;

		CSimpleKroneckerLMM lmm;
		lmm.setK_R(K);
		lmm.setK_C(Kp);
		lmm.setPheno(pheno);
		lmm.setSNPs(snps);
		lmm.setCovs(covs);
		lmm.setWkron(Wkron);
		lmm.setWkron0(Wkron0);
		lmm.process();
	}

	if (1){ //LMM testing using new code
		CLMM lmm;


		VectorXd v = VectorXd::Ones(3);
		MatrixXd M = MatrixXd::Ones(3,2);

		isnull(v);
		isnull(v.transpose());

		isnull(M);

		lmm.setK(K);
		lmm.setSNPs(snps);
		lmm.setPheno(pheno);
		lmm.setCovs(covs);
		lmm.setTestStatistics(lmm.TEST_F);

		lmm.process();
		MatrixXd pv = lmm.getPv();
		cout <<"pv_new:\n"<< scientific <<pv<<endl;
	}




	if(0) //kronecker product LMM
	{
		//TODO: calculate dofs for arbitrary WkronDiag and WkronBlock, currently we expect all ones...
		MatrixXd WkronDiag0, WkronBlock0, WkronDiag, WkronBlock;
		if (0)
		{//one weight per SNP over all phenotypes
			WkronDiag0=MatrixXd::Ones(1,ncov);
			WkronBlock0=MatrixXd::Ones(p,ncov);
			WkronDiag=MatrixXd::Ones(1,ncov+1);
			WkronBlock=MatrixXd::Ones(p,ncov+1);
		}
		else
		{//phenotype many weights per SNP
			WkronDiag0=MatrixXd::Ones(p,ncov);
			WkronBlock0=MatrixXd::Ones(1,ncov);
			WkronDiag=MatrixXd::Ones(p,ncov+1);
			WkronBlock=MatrixXd::Ones(1,ncov+1);
		}


		MatrixXd Kp = 1.0/p * (snps.block(0,0,p,s)*snps.block(0,0,p,s).transpose());
		//cout <<Kp;
		CKroneckerLMM kron;
		kron.setKronStructure(WkronDiag0, WkronBlock0, WkronDiag, WkronBlock);
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
#endif
