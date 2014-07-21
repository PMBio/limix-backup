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
#include "limix/gp/gp_kronecker.h"
#include "limix/utils/matrix_helper.h"
#include "limix/covar/linear.h"
#include "limix/covar/se.h"
#include "limix/covar/fixed.h"
#include "limix/covar/combinators.h"
#include "limix/mean/CData.h"
#include "limix/mean/CKroneckerMean.h"
#include "limix/utils/logging.h"
#include <vector>


using namespace std;
using namespace limix;
#ifndef PI
#define PI 3.14159265358979323846
#endif

#define GPLVM

int main() {
	bool useIdentity = false;

	std::vector<Pbool> test;

	Pbool a;
	test.push_back(a);

	//try {
		//random input X
		muint_t Kr=2;
		muint_t Kc=3;

		muint_t D=20;
		muint_t N=20;

		if (useIdentity)
		{
			Kr = N;
			Kc = D;
		}
		mfloat_t eps = 0.1;

		//"simulation"
		MatrixXd X = randn((muint_t)N,(muint_t)Kr);
		//y ~ w*X
		MatrixXd w = randn((muint_t)Kr,(muint_t)D);
		MatrixXd y = X*w + eps*randn((muint_t)N,(muint_t)D);

#if 0
		MatrixXd Xr = randn(N,Kr);
		//covariances
		PCovLinearISO covar_r(new CCovLinearISO(Kr));
#else	//identity for rows
		//use simple fixed covarainces: identities on rows and colmns
		MatrixXd Mr = MatrixXd::Identity(N,N);
		MatrixXd Xr = MatrixXd::Zero(N,0);
		sptr<CFixedCF> covar_r(new CFixedCF(Mr));
#endif
#if 0
		MatrixXd Xc = randn(D,Kc);
		//covariances
		PCovLinearISO covar_c1(new CCovLinearISO(Kc));
		PEyeCF covar_c2(new CEyeCF());

		PSumCF covar_c(new CSumCF());
		covar_c->addCovariance(covar_c1);
		covar_c->addCovariance(covar_c2);

#else	//identity for cols
		//use simple fixed covarainces: identities on rows and colmns
		MatrixXd Mc = MatrixXd::Identity(D,D);
		sptr<CFixedCF> covar_c(new CFixedCF(Mc));
		//inputs are fake inputs
		MatrixXd Xc = MatrixXd::Zero(D,0);
#endif
		//likelihood
		PLikNormalSVD lik(new CLikNormalSVD());

		//Data term
		MatrixXd A = MatrixXd::Ones(1,D);
		MatrixXd fixedEffects = MatrixXd::Ones(N,1);
		MatrixXd weights = 0.5+MatrixXd::Zero(1,1).array();
		PKroneckerMean data(new CKroneckerMean(y,weights,fixedEffects,A));

		//hyperparams: scaling parameters of covariance functions
		CovarInput covar_params_r = MatrixXd::Zero(covar_r->getNumberParams(),1);
		CovarInput covar_params_c = MatrixXd::Zero(covar_c->getNumberParams(),1);

		//GP object
		PGPkronecker gp(new CGPkronecker(covar_r,covar_c,lik,data));
		gp->setX_r(Xr);
		gp->setX_c(Xc);
		gp->setY(y);


		CovarInput lik_params = randn(lik->getNumberParams(),1);
		//lik_params(0) = 0.5*log(1);
		lik_params(1) = 0.1;
		CGPHyperParams params;
		params["covar_r"] = covar_params_r;
		params["covar_c"] = covar_params_c;
		params["lik"] = lik_params;
		params["dataTerm"] = weights;
		params["X_r"] = Xr;
		params["X_c"] = Xc;

		//set full params for initialization
		gp->setParams(params);

		//simplify optimization: remove covar_r, covar_c, lik
		CGPHyperParams opt_params(params);
		//opt_params.erase("lik");
		//opt_params.erase("covar_r");
		//opt_params.erase("covar_c");
		//opt_params.erase("X_r");
		//opt_params.erase("dataTerm");
		//opt_params.erase("X_c");

		std::cout << gp->LML() << "\n";
		std::cout << gp->LML() << "\n";

		params["covar_r"](0) = 5;
		gp->setParams(params);
		std::cout << gp->LML() << "\n";



#if 1
		CGPopt opt(gp);
		opt.addOptStartParams(opt_params);
		CGPHyperParams upper;
		CGPHyperParams lower;
		upper["lik"] = 5.0*MatrixXd::Ones(2,1);
		lower["lik"] = -5.0*MatrixXd::Ones(2,1);
		opt.setOptBoundLower(lower);
		opt.setOptBoundUpper(upper);

		std::cout << "gradcheck"
				": "<< opt.gradCheck()<<"\n";
		//optimize:
		opt.opt();

		MatrixXd out;
		gp->apredictMean(&out,Xr,Xc);

		std::cout << "=====post opt=====" << "\n";
		std::cout << "lml("<<gp->getParams()<<")=" <<gp->LML()<< "\n";
		std::cout << "dlml("<<gp->getParams()<<")=" <<gp->LMLgrad()<< "\n";
		std::cout << "==========" << "\n";

		std::cout << "gradcheck: "<< opt.gradCheck()<<"\n";
#endif

	//}
	//catch(CLimixException& e) {
	//	cout <<"Exception : "<< e.what() << endl;
	//}

}

#endif
