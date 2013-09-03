//============================================================================
// Name        : GPmix.cpp
// Author      :
// Version     :
// Copyright   : Your copyright notice
// Test file for kronecker LMM test
//============================================================================

#if 1

#include <iostream>
#include "limix/types.h"
#include "limix/LMM/kronecker_lmm.h"

using namespace std;
using namespace limix;
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


		}
		catch(CGPMixException& e) {
			cout <<"Exception : "<< e.what() << endl;
		}

}

#endif
