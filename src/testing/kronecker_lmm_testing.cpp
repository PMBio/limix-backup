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
#include "gpmix/LMM/kronecker_lmm.h"

using namespace std;
using namespace gpmix;
#ifndef PI
#define PI 3.14159265358979323846
#endif

#define GPLVM

int main() {

	try{
		//random input X
		muint_t Wr=1;
		muint_t Wc=1;

		muint_t D=200;
		muint_t N=20;

		mfloat_t eps = 100.0;

		//1. "simulation"
		MatrixXd X = randn((muint_t)N,(muint_t)Wr);
		//y ~ w*X
		MatrixXd w = randn((muint_t)Wr,(muint_t)Wc);

		MatrixXd A = MatrixXd::Ones((muint_t)Wc,(muint_t)D);
		MatrixXd y = X*w*A + 1.0*eps*randn((muint_t)N,(muint_t)D);
		//SNPS: all random except for one true causal guy

		MatrixXd S = MatrixXd::Zero((muint_t)N,20);
		S.block(0,5,N,10) = randn((muint_t)N,10);
		S.block(0,4,N,Wr) = X;
		//std::cout<<"SNPS: "<< S<<std::endl;
		//2. construction of GP object
		muint_t Kr=3;
		muint_t Kc=4;
#if 1
		MatrixXd Xr = randn(N,Kr);
		//covariances
		PCovLinearISO covar_r(new CCovLinearISO(Kr));
#else	//identity for rows
		//use simple fixed covariances: identities on rows and colmns
		MatrixXd Mr = MatrixXd::Identity(N,N);
		MatrixXd Xr = MatrixXd::Zero(N,0);
		sptr<CFixedCF> covar_r(new CFixedCF(Mr));
#endif
#if 1
		MatrixXd Xc = randn(D,Kc);
		//covariances
		PCovLinearISO covar_c(new CCovLinearISO(Kc));
#else	//identity for cols
		//use simple fixed covariances: identities on rows and columns
		MatrixXd Mc = MatrixXd::Identity(D,D);
		sptr<CFixedCF> covar_c(new CFixedCF(Mc));
		//inputs are fake inputs
		MatrixXd Xc = MatrixXd::Zero(D,0);
#endif


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
		//params["covar_r"] = covar_params_r;
		//params["covar_c"] = covar_params_c;
		params["lik"] = lik_params;

		//set full params for initialization
		gp->setParams(params);

		//simplify optimizatin: remove covar_r, covar_c, lik
		CGPHyperParams opt_params(params);
		//opt_params.erase("lik");
		opt_params.erase("covar_r");
		opt_params.erase("covar_c");
		opt_params.erase("X_r");
		//opt_params.erase("dataTerm");
		opt_params.erase("X_c");

		double lml = gp->LML();
		std::cout << lml<<endl;

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
		CKroneckerLMM lmm_(gp);
		//set SNPs
		lmm_.setSNPs(S);
		//set Phenotypes
		lmm_.setPheno(y);
		//set covariates
		lmm_.setCovs(MatrixXd::Ones(X.rows(),1));
		//set design matrics: both testing all genes
		MatrixXd A_ = MatrixXd::Ones(1,D);
		MatrixXd A0_= MatrixXd::Ones(1,D);
		lmm_.setAAlt(A_);
		lmm_.addA0(A0_);
		std::cout << "Start CKroneckerLMM:" << "\n";
		lmm_.process();
		MatrixXd pv_ = lmm_.getPv();
		//std::cout << pv_ << "\n" << "   0.99032           1    0.999952    0.995441   0.0277248 0.000805652    0.999212           1    0.999999    0.600988bla" << "\n";

#endif


#if 0
		//test CGPLMM
		CGPLMM lmm(gp);
		//set SNPs
		lmm.setSNPs(S);
		//set Phenotypes
		lmm.setPheno(y);
		//set covariates
		lmm.setCovs(MatrixXd::Ones(X.rows(),1));
		//set design matrics: both testing all genes
		MatrixXd AAlt = MatrixXd::Ones(1,D);
		MatrixXd A0= MatrixXd::Ones(1,D);
		lmm.setAAlt(AAlt);
		lmm.addA0(A0);
		std::cout << "Start:" << "\n";
		lmm.process();
		MatrixXd pv = lmm.getPv();
		//std::cout << pv << "\n" << "   0.99032           1    0.999952    0.995441   0.0277248 0.000805652    0.999212           1    0.999999    0.600988bla" << "\n";

#endif


		}
		catch(CGPMixException& e) {
			cout <<"Exception : "<< e.what() << endl;
		}

}

#endif
