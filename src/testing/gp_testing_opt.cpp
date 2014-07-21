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
#include "limix/gp/gp_base.h"
#include "limix/gp/gp_opt.h"
#include "limix/types.h"
#include "limix/likelihood/likelihood.h"
#include "limix/gp/gp_base.h"
#include "limix/utils/matrix_helper.h"
#include "limix/covar/linear.h"
#include "limix/covar/se.h"
#include "limix/covar/fixed.h"
#include "limix/covar/combinators.h"
#include "limix/covar/freeform.h"
#include "limix/mean/CLinearMean.h"
#include "limix/mean/CKroneckerMean.h"
#include "limix/mean/CSumLinear.h"
#include "limix/mean/CData.h"

using namespace std;
using namespace limix;
#ifndef PI
#define PI 3.14159265358979323846
#endif

int main() {

	muint_t N = 100;
	muint_t S = 1000;
	muint_t G = 500;
	muint_t Ne = 2;

	MatrixXd Y = randn((muint_t)N,(muint_t)G);
	MatrixXd X = randn((muint_t)N,(muint_t) S);
	MatrixXd Kpop = 1.0/S * X*X.transpose();
	MatrixXd E=MatrixXd::Zero(N,1);

	MatrixXd XE = MatrixXd::Zero(N,2);

	CSumCF covar;

	CFixedCF CG1(Kpop);
	CCovFreeform CG2(Ne);
	CProductCF CG;
	CG.addCovariance(&CG1);
	CG.addCovariance(&CG2);
	CCovFreeform CE(Ne);

	covar.addCovariance(&CG);
	covar.addCovariance(&CE);


	CLikNormalIso lik;
	CData data;
	CGPHyperParams params,params2;


	params["covar"] = randn(covar.getNumberParams(),(muint_t)1);
	params["lik"] = randn(lik.getNumberParams(),(muint_t)1);




	CGPbase gp(data, covar, lik);
	gp.setY(Y);
	gp.setX(XE);
	gp.setParams(params);

	CGPopt opt(gp);
	std::cout << "gradcheck: "<< opt.gradCheck() << "\n";

	//optimize:
	//construct constraints
	CGPHyperParams upper;
	CGPHyperParams lower;
	upper["lik"] = 5.0*MatrixXd::Ones(1,1);
	lower["lik"] = -5.0*MatrixXd::Ones(1,1);
	opt.setOptBoundLower(lower);
	opt.setOptBoundUpper(upper);
	//set start ponts
	opt.addOptStartParams(params);

	//create filter
	CGPHyperParams filter;
	MatrixXd filter_covar =  MatrixXd::Ones(params["covar"].rows(),1);
	filter_covar(0) = 0.0;
	filter["covar"] = filter_covar;
	//filter["lik"] = MatrixXd::Ones(params["lik"].rows(),1);


	opt.setParamMask(filter);
	opt.opt();

	std::cout << "\n\n-----------------\n";

	//print stuff
	CGPHyperParams lmlgrad = gp.LMLgrad();
	//1. gradients at optimum:
	std::cout << lmlgrad["covar"] << "," << lmlgrad["lik"]<< "\n";
	//2. changes of parameters
	std::cout << "params0:"<< "\n"<< params << "\n";
	std::cout << "paramsO:"<< "\n"<< gp.getParams() << "\n";


}



#endif

