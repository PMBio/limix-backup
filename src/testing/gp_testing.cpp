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

//#define linMean
//#define kronMean
//#define sumKronMean

int main() {

#if 0

	muint_t N = 100;
	muint_t S = 1000;
	muint_t G = 500;
	muint_t Ne = 5;

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
	CGPHyperParams params;
	CovarInput covar_params = randn(covar.getNumberParams(),(muint_t)1);
	CovarInput lik_params = randn(lik.getNumberParams(),(muint_t)1);

	params["covar"] = covar_params;
	params["lik"] = lik_params;



	CGPbase gp(data, covar, lik);
	gp.setY(Y);
	gp.setX(XE);
	gp.setParams(params);
	std::cout <<"grad[covar] :"<< params["covar"] << "\n";
	std::cout <<"grad[lik] :"<< params["lik"] << "\n";
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

	opt.opt();

#endif

	try {
		//random input X
		muint_t dim=2;
		muint_t nsamples =100;
		muint_t targets = 1;
		muint_t nWeights = 2;

		MatrixXd X = randn(nsamples,dim);
		//y ~ w*X
		MatrixXd w = randn(dim,nWeights);
		MatrixXd A = randn(nWeights, targets);
		MatrixXd y = X*w*A + 0.1*randn(nsamples,targets);

		CGPHyperParams params;


		MatrixXd cov = MatrixXd::Ones(nsamples,1);
		PLinearMean mean(new CLinearMean(y,cov));
		PLikNormalIso lik(new CLikNormalIso());


		MatrixXd weights = MatrixXd::Ones(1,1);


		//Ard covariance
		PCovLinearARD covar(new CCovLinearARD(X.cols()));
		//GP object
		PGPbase gp(new CGPbase(covar,lik,mean));
		gp->setY(y);
		gp->setX(X);
		//hyperparams
		CovarInput covar_params = randn(gp->getCovar()->getNumberParams(),(muint_t)1);
		CovarInput lik_params = randn(gp->getLik()->getNumberParams(),(muint_t)1);
		params["covar"] = covar_params;
		params["lik"] = lik_params;
		params["dataTerm"] = weights;
		gp->setParams(params);


		std::cout << gp->LML() << "\n";
		std::cout << gp->LML() << "\n";

		params["lik"](0) = -2;
		gp->setParams(params);
		std::cout << gp->LML() << "\n";


#if 1
		mfloat_t lml = gp->LML(params);
		CGPHyperParams grad = gp->LMLgrad();
		std::cout <<"lml : "<< lml << "\n";
		std::cout <<"grad :"<< grad << "\n";
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
		opt.opt();
#endif


		gp->predictMean(X.block(0,0,10,X.cols()));
		gp->predictVar(X.block(0,0,10,X.cols()));



	}
	catch(CLimixException& e) {
		cout << e.what() << endl;
	}


}



#endif

