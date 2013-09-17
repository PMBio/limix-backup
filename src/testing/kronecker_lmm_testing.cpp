//============================================================================
// Name        : GPmix.cpp
// Author      :
// Version     :
// Copyright   : Your copyright notice
// Test file for kronecker LMM test
//============================================================================

#if 1

#define debugkron 1

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
		muint_t Wr_covar = 2;
		muint_t Wc_covar = 1;
		muint_t P=2;
		muint_t N=200;
		muint_t Wc=P;
		
		MatrixXd Xr1=randn(N,N)/sqrt(N);
		MatrixXd Kr1=Xr1*Xr1.transpose()/N;

		MatrixXd Xc1=randn(P,P)/sqrt(P);
		MatrixXd Kc1=Xc1.transpose()*Xc1;
		
		MatrixXd Xr2 = MatrixXd::Identity(N,N);
		MatrixXd Kr2 = Xr2*Xr2.transpose();
		MatrixXd Xc2=randn(P,P)/sqrt(P);
		MatrixXd Kc2 = Xc2.transpose()*Xc2;

		mfloat_t eps1 = 1.0;
		mfloat_t eps2 = 1.0;

		//1. "simulation"
		MatrixXd X = randn((muint_t)N,(muint_t)Wr);
		MatrixXd Xcovar = randn((muint_t)N,(muint_t)Wr_covar);
		//y ~ w*X
		MatrixXd w = randn((muint_t)Wr,(muint_t)Wc);
		MatrixXd wcovar = randn((muint_t)Wr_covar,(muint_t)Wc_covar);

		MatrixXd A = MatrixXd::Identity((muint_t)Wc,(muint_t)P);
		MatrixXd A_covar = MatrixXd::Ones((muint_t)Wc_covar,(muint_t)P);
		MatrixXd noise1 = Xr1*1.0*eps1*randn((muint_t)N,(muint_t)P)*Xc1;
		MatrixXd noise2 = Xr2*1.0*eps2*randn((muint_t)N,(muint_t)P)*Xc2;
		MatrixXd Y = X*w*A + Xcovar*wcovar*A_covar + noise1 + noise2;
		//SNPS: all random except for one true causal guy

		muint_t Nsnp0=50;
		MatrixXd S = MatrixXd::Zero((muint_t)N,(Nsnp0+1)*Wr);
		S.block(0,0,N,Nsnp0) = randn((muint_t)N,Nsnp0*Wr);
		S.block(0,Nsnp0*Wr,N,Wr) = X;
		//std::cout<<"SNPS: "<< S<<std::endl;
		//2. construction of GP object

		MatrixXdVec Acov=MatrixXdVec();
		Acov.push_back(A_covar);
		MatrixXdVec Xcov=MatrixXdVec();
		Xcov.push_back(Xcovar);

		CKroneckerLMM lmm = CKroneckerLMM();
		lmm.setK1c(Kc1);
		lmm.setK1r(Kr1);
		lmm.setK2c(Kc2);
		lmm.setK2r(Kr2);
		lmm.setSNPs(S);
		lmm.setSNPcoldesign(A);
		lmm.setCovariates(Xcov,Acov);
		lmm.setPheno(Y);
		lmm.setNumIntervalsAlt(0);//TODO: nLLeval with delta unequal to 1 does not work properly yet
		lmm.setNumIntervals0(0);//TODO: nLLeval with delta unequal to 1 does not work properly yet 
		lmm.process();
		MatrixXd pv = MatrixXd();
		lmm.agetPv(&pv);
		std::cout << "pv:\n" << pv << "\n";
		std::cout << "done.\n" ;
		
		}
		catch(CGPMixException& e) {
			cout <<"Exception : "<< e.what() << endl;
		}

}

#endif
