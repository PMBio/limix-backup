#if 0
//============================================================================
// Name        : GPmix.cpp
// C++ testig file for variance component QTL
//============================================================================

#include <iostream>
#include "limix/types.h"
#include "limix/covar/covariance.h"
#include "limix/covar/linear.h"
#include "limix/utils/matrix_helper.h"
#include "limix/modules/CMultiTraitVQTL.h"
#include "limix/modules/CVarianceDecomposition.h"


using namespace limix;

using namespace std;
using namespace limix;

int main() {

	//1. simulate
	muint_t n  = 20;
	muint_t p = 2;
	muint_t s = 1000;
	muint_t ncov = 1;

	MatrixXd snps = (MatrixXd)randn((muint_t)n,(muint_t)s);
	MatrixXd pheno = (MatrixXd)randn((muint_t)n,(muint_t)p);
	MatrixXd covs = MatrixXd::Ones(n,ncov);

	//1. merge phenotypes, they are concatenated
	MatrixXd X = MatrixXd::Zero(p*n,s);
	X.block(0,0,n,s) = snps;
	X.block(n,0,n,s) = snps;
	MatrixXd Y = MatrixXd::Zero(p*n,1);
	Y.block(0,0,n,1) = pheno.block(0,0,n,1);
	Y.block(n,0,n,1) = pheno.block(0,1,n,1);
	MatrixXd C = MatrixXd::Zero(p*n,p);
	C.block(0,0,n,1) = MatrixXd::Ones(n,1);
	C.block(n,1,n,1) = MatrixXd::Ones(n,1);
	MatrixXd T = 5*MatrixXd::Ones(p*n,1);
	T.block(0,0,n,1) = 4*MatrixXd::Ones(n,1);


	//2. get K matrix
	MatrixXd K = 1.0/X.cols() * (X*X.transpose());
	//3. genotype identitiy matrix
	MatrixXd Kgeno = MatrixXd::Zero(p*n,p*n);
	for (muint_t i=0;i<n;++i)
	{
		Kgeno(i,i) = 1;
		Kgeno(i,i+n) = 1;
		Kgeno(i+n,i) = 1;
		Kgeno(i+n,i+n) = 1;
	}


	//3. construct object and run infernece
	PMultiTraitVQTL mqtl(new CMultiTraitVQTL());
	mqtl->addK(K);
	mqtl->setKgeno(Kgeno);
	mqtl->setTrait(T);
	mqtl->setPheno(Y);
	mqtl->setFixed(C);


	//train
	mqtl->train();
	//get variance of a covar term & noise
	MatrixXd Vnoise = mqtl->getVarianceComponent_noise();
	MatrixXd Vterm = mqtl->getVarianceComponent_term(0);

	std::cout << Vnoise << "\n";
	std::cout << Vterm << "\n";


	//4. new model
	PVarianceDecomposition vdc(new CVarianceDecomposition(Y,T));
	vdc->setFixed(C);
	vdc->addTerm(K,CVarianceDecomposition::categorial,0.2,true);
	vdc->addTerm(Kgeno,CVarianceDecomposition::categorial,0.8,false);

	vdc->train();





}

#endif
