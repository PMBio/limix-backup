//============================================================================
// Name        : GPmix.cpp
// Author      :
// Version     :
// Copyright   : Your copyright notice
// Test file for kronecker LMM test
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
#include "gpmix/LMM/CGPLMM.h"

using namespace std;
using namespace gpmix;
#ifndef PI
#define PI 3.14159265358979323846
#endif

#define GPLVM

int main() {

		//random input X
		muint_t Kr=2;
		muint_t Kc=3;

		muint_t D=10;
		muint_t N=20;

		mfloat_t eps = 0.1;

		//1. "simulation"
		MatrixXd X = randn((muint_t)N,(muint_t)Kr);
		//y ~ w*X
		MatrixXd w = randn((muint_t)Kr,(muint_t)D);
		MatrixXd y = X*w + eps*randn((muint_t)N,(muint_t)D);
		//SNPS: all random except for one true causal guy
		MatrixXd S = randn((muint_t)N,10);
		S.block(0,4,N,Kr) = X;



		//2. consturction of GP object
		//use identity matrices; in all other cases there is still a bug!
		Kc = D;
		Kr = N;

		//use simple fixed covarainces: identities on rows and colmns
		sptr<CFixedCF> covar_r(new CFixedCF(MatrixXd::Identity(N,N)));
		sptr<CFixedCF> covar_c(new CFixedCF(MatrixXd::Identity(D,D)));
		//inputs are fake inputs
		MatrixXd Xr = MatrixXd::Zero(N,0);
		MatrixXd Xc = MatrixXd::Zero(D,0);

		//hyperparams: scalig parameters of covariace functions
		CovarInput covar_params_r = MatrixXd::Zero(covar_r->getNumberParams(),1);
		CovarInput covar_params_c = MatrixXd::Zero(covar_c->getNumberParams(),1);

		//GP object
		sptr<CGPkronecker> gp(new CGPkronecker(covar_r,covar_c));
		gp->setX_r(Xr);
		gp->setX_c(Xc);
		//gp->setDataTerm(data);
		gp->setY(y);


		CovarInput lik_params = randn(gp->getLik()->getNumberParams(),1);
		CGPHyperParams params;
		params["covar_r"] = covar_params_r;
		params["covar_c"] = covar_params_c;
		params["lik"] = lik_params;

		//set full params for initialization
		gp->setParams(params);

		//simplify optimizatin: remove covar_r, covar_c, lik
		CGPHyperParams opt_params(params);
		//opt_params.erase("lik");
		//opt_params.erase("covar_r");
		//opt_params.erase("covar_c");
		//opt_params.erase("X_r");
		//opt_params.erase("dataTerm");
		//opt_params.erase("X_c");

		//double lml = gp.LML();

		//set restricted param object without lik, covar_r, covar_c:
		gp->setParams(opt_params);
		CGPopt opt(gp);


#if 0
		opt_params["dataTerm"](0) =2.2;
		gp->setParams(opt_params);
		std::cout << "lmlgrad("<<opt_params << "):\n";
		std::cout << gp->LMLgrad() << "\n";

		opt_params["dataTerm"](0) =1.0;
		gp->setParams(opt_params);
		std::cout << "lmlgrad("<<opt_params << "):\n";
		std::cout << gp->LMLgrad() << "\n";
#endif


#if 0
		CGPHyperParams upper;
		CGPHyperParams lower;
		upper["lik"] = 5.0*MatrixXd::Ones(1,1);
		lower["lik"] = -5.0*MatrixXd::Ones(1,1);
		opt.setOptBoundLower(lower);
		opt.setOptBoundUpper(upper);

		std::cout << "gradcheck"
				": "<< opt.gradCheck()<<"\n";
		//optimize:
		opt.opt();

		std::cout << "=====post opt=====" << "\n";
		std::cout << "lml("<<gp->getParams()<<")=" <<gp->LML()<< "\n";
		std::cout << "dlml("<<gp->getParams()<<")=" <<gp->LMLgrad()<< "\n";
		std::cout << "==========" << "\n";
#endif

#if 0
		std::cout << "gradcheck"
						": "<< opt.gradCheck()<<"\n";

		//Data term
		MatrixXd A = MatrixXd::Ones(1,D);
		MatrixXd fixedEffects = MatrixXd::Ones(N,1);
		MatrixXd weights = 0.5+MatrixXd::Zero(1,1).array();
		PKroneckerMean data(new CKroneckerMean(y,weights,fixedEffects,A));
		opt_params["dataTerm"] = weights;
		gp->setParams(opt_params);
		gp->setDataTerm(data);

		std::cout << "gradcheck"
					": "<< opt.gradCheck()<<"\n";



#endif


#if 1
		//test CGPLMM
		CGPLMM lmm(gp);
		//set SNPs
		lmm.setSNPs(S);
		//set Phenotypes
		lmm.setPheno(y);
		//set covariates
		lmm.setCovs(MatrixXd::Ones(X.rows(),1));
		//set design matrics: both testing all genes
		MatrixXd A = MatrixXd::Ones(2,D);
		MatrixXd A0= MatrixXd::Ones(1,D);
		lmm.setA(A);
		lmm.setA0(A0);
		std::cout << "True SNPs are 4 and 5, so looks like this is somewhat working..." << "\n";
		lmm.process();
		MatrixXd pv = lmm.getPv();
#endif



}

#endif
