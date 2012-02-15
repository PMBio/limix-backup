//============================================================================
// Name        : GPmix.cpp
// Author      :
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#if 1

#include <iostream>
#include "gpmix/gp/gp_base.h"
#include "gpmix/gp/gp_opt.h"
#include "gpmix/types.h"
#include "gpmix/likelihood/likelihood.h"
#include "gpmix/gp/gp_base.h"
#include "gpmix/gp/gp_kronecker.h"
#include "gpmix/utils/matrix_helper.h"
#include "gpmix/covar/linear.h"
#include "gpmix/covar/se.h"
#include "gpmix/covar/fixed.h"
#include "gpmix/covar/combinators.h"
#include "gpmix/mean/CData.h"
#include "gpmix/mean/CKroneckerMean.h"
#include "gpmix/utils/logging.h"


using namespace std;
using namespace gpmix;
#ifndef PI
#define PI 3.14159265358979323846
#endif

#define GPLVM

int main() {
	bool useIdentity = false;

	try {
		//random input X
		muint_t Kr=2;
		muint_t Kc=3;

		muint_t D=1;
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




		//Kc = D;
		//Kr = N;



#if 1
		MatrixXd Xr = randn(N,Kr);
		//covariances
		PCovLinearISO covar_r(new CCovLinearISO(Kr));
#else	//identity for rows
		//use simple fixed covarainces: identities on rows and colmns
		MatrixXd Mr = MatrixXd::Identity(N,N);
		MatrixXd Xr = MatrixXd::Zero(N,0);
		sptr<CFixedCF> covar_r(new CFixedCF(Mr));
#endif
#if 1
		MatrixXd Xc = randn(D,Kc);
		//covariances
		PCovLinearISO covar_c(new CCovLinearISO(Kc));
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
		CGPHyperParams params;
		params["covar_r"] = covar_params_r;
		params["covar_c"] = covar_params_c;
		params["lik"] = lik_params;
		params["dataTerm"] = weights;

		//set full params for initialization
		gp->setParams(params);

		//simplify optimization: remove covar_r, covar_c, lik
		CGPHyperParams opt_params(params);
		opt_params.erase("lik");
		//opt_params.erase("covar_r");
		//opt_params.erase("covar_c");
		//opt_params.erase("X_r");
		//opt_params.erase("dataTerm");
		//opt_params.erase("X_c");

		//double lml = gp.LML();

		//set restricted param object without lik, covar_r, covar_c:
		gp->setParams(opt_params);

#if 0
		opt_params["dataTerm"](0) =4.2;
		gp->setParams(opt_params);
		std::cout << "lmlgrad("<<opt_params << "):\n";
		std::cout << gp->LML()<< "---" << gp->LMLgrad() << "----\n";

		opt_params["dataTerm"](0) =1.0;
		gp->setParams(opt_params);
		std::cout << "lmlgrad("<<opt_params << "):\n";
		std::cout << gp->LML()<< "---" << gp->LMLgrad() << "----\n";
#endif

#if 1
		/*
		std::cout <<"grad" << grad << "\n";
		std::cout << "=====pre opt=====" << "\n";
		std::cout << "lml("<<gp.getParams()<<")=" <<gp.LML()<< "\n";
		std::cout << "dlml("<<gp.getParams()<<")=" <<gp.LMLgrad()<< "\n";
		std::cout << "==========" << "\n";
		*/

		//std::cout << gp.getCache().cache_c.getK0() << "\n";
		//std::cout << gp.getCache().cache_c.getUK() << "\n";

		CGPopt opt(gp);
		CGPHyperParams upper;
		CGPHyperParams lower;
		upper["lik"] = 5.0*MatrixXd::Ones(1,1);
		lower["lik"] = -5.0*MatrixXd::Ones(1,1);
		opt.setOptBoundLower(lower);
		opt.setOptBoundUpper(upper);

		std::cout << "gradcheck"
				": "<< opt.gradCheck()<<"\n";
		//optimize:
		//opt.opt();

		std::cout << "=====post opt=====" << "\n";
		std::cout << "lml("<<gp->getParams()<<")=" <<gp->LML()<< "\n";
		std::cout << "dlml("<<gp->getParams()<<")=" <<gp->LMLgrad()<< "\n";
		std::cout << "==========" << "\n";

		std::cout << "gradcheck: "<< opt.gradCheck()<<"\n";
#endif

	}
	catch(CGPMixException& e) {
		cout <<"Exception : "<< e.what() << endl;
	}


}

#endif
