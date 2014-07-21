// Copyright(c) 2014, The LIMIX developers(Christoph Lippert, Paolo Francesco Casale, Oliver Stegle)
//
//Licensed under the Apache License, Version 2.0 (the "License");
//you may not use this file except in compliance with the License.
//You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
//Unless required by applicable law or agreed to in writing, software
//distributed under the License is distributed on an "AS IS" BASIS,
//WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//See the License for the specific language governing permissions and
//limitations under the License.

#if 0


#include <iostream>
#include "limix/types.h"
#include "limix/covar/covariance.h"
#include "limix/covar/linear.h"
#include "limix/utils/matrix_helper.h"
#include "limix/LMM/lmm.h"
#include "limix/LMM/lmm_old.h"
using namespace limix;

using namespace std;
using namespace limix;

int main() {

	int n = 100;
	int p = 1;
	int s = 10;
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



	if (1){ //LMM testing using new code
		CLMM lmm;


		VectorXd v = VectorXd::Ones(3);
		MatrixXd M = MatrixXd::Ones(3,2);

		lmm.setK(K);
		lmm.setSNPs(snps);
		lmm.setPheno(pheno);
		lmm.setCovs(covs);
		lmm.process();


		lmm.setSNPs(snps);
		lmm.process();



		MatrixXd pv = lmm.getPv();
		cout <<"pv:\n"<< scientific <<pv<<endl;

		CInteractLMM ilmm;
		ilmm.setK(K);
		ilmm.setSNPs(snps);
		ilmm.setPheno(pheno);
		ilmm.setCovs(covs);
		MatrixXd Im = MatrixXd::Ones(snps.rows(),1);
		Im.block(0,0,50,1).setConstant(0.0);
		ilmm.setInter(Im);
		ilmm.setInter0(MatrixXd::Ones(snps.rows(),1));
		ilmm.process();
		MatrixXd ipv = ilmm.getPv();


		ilmm.setTestStatistics(ilmm.TEST_F);
		ilmm.process();
		MatrixXd ipv2 = ilmm.getPv();

		cout <<"ipv:\n"<< scientific <<ipv<<endl;
		cout <<"ipv2:\n"<< scientific <<ipv2<<endl;


		//std::cout << pv-ipv << "\n";
	}


}
#endif
