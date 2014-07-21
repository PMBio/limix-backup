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

	/*

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
	*/

	//4. new model
	PVarianceDecomposition vdc(new CVarianceDecomposition(Y,T));
	vdc->setFixed(C);

	//initialize 3 terms
	PCategorialTraitVarianceTerm term0(new CCategorialTraitVarianceTerm(K,T));
	term0->setConstraint(freeform);
	term0->setVarianceInit(0.1);

	PCategorialTraitVarianceTerm term1(new CCategorialTraitVarianceTerm(K,T));
	term1->setConstraint(freeform);

	PCategorialTraitVarianceTerm noise(new CCategorialTraitVarianceTerm(K,T));
	noise->setConstraint(diagonal);
	term0->setVarianceInit(0.01);


	vdc->addTerm(term0);
	//vdc->addTerm(term1);
	vdc->addTerm(noise);


	//init GP
	vdc->initGP();

	std::cout<< term0->getCovariance()->getParamMask() << "\n\n";
	std::cout << noise->getCovariance()->getParamMask() << "\n\n";

	std::cout << vdc->getGP()->getCovar()->getParamMask() << "\n\n";

	std::cout << vdc->getGP()->getParams() << "\n\n";



	PGPbase gp = vdc->getGP();

	vdc->train();

	term0->getVariance();





}

#endif
