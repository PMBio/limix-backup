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

//#define debugkron 1

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
#ifndef debugkron
		muint_t Wr=1;
		muint_t Wr_covar = 2;
		muint_t Wc_covar = 1;
		muint_t P=50;
		muint_t N=100;
		muint_t Wc=P;
		muint_t Nsnp0=5000;
		muint_t NsnpK=500;
		mfloat_t eps1 = 1.0;
		mfloat_t eps2 = 1.0;
#else
		muint_t Wr=1;
		muint_t Wr_covar = 2;
		muint_t Wc_covar = 1;
		muint_t P=2;
		muint_t N=3;
		muint_t Wc=P;
		muint_t Nsnp0=50;
		mfloat_t eps1 = 1.0;
		mfloat_t eps2 = 2.0;
#endif
		//1. "simulation"
		MatrixXd Xr1=BaldingNichols((muint_t)N,NsnpK, 0.1, 0.005, true);
		Xr1/=limix::sqrt((mfloat_t)(NsnpK));
		MatrixXd Kr1=Xr1*Xr1.transpose();//+MatrixXd::Identity(N,N);
		if (Kr1.hasNaN()){
			std::cout << Kr1;
		}
		MatrixXd Xc1=randn(2*P,P);
		Xc1/=limix::sqrt((mfloat_t)(2.0*P));
		MatrixXd Kc1=Xc1.transpose()*Xc1;//+MatrixXd::Identity(P,P);
		if (Kc1.hasNaN()){
			std::cout << Kc1;
		}
		MatrixXd Xr2 = BaldingNichols((muint_t)N,2*N, 0.1, 0.005, true);
		Xr2/=limix::sqrt((mfloat_t)(2.0*N));
		MatrixXd Kr2 = Xr2*Xr2.transpose();//+MatrixXd::Identity(N,N);
		if (Kr2.hasNaN()){
			std::cout << Kr2;
		}
		MatrixXd Xc2=randn(2*P,P);
		Xc2/=limix::sqrt((mfloat_t)(2.0*P));
		//Xc2 = MatrixXd::Identity(P,P);
		MatrixXd Kc2 = Xc2.transpose()*Xc2;//+MatrixXd::Identity(P,P);
		if (Kc2.hasNaN()){
			std::cout << Kc2;
		}

		
		MatrixXd X = BaldingNichols((muint_t)N,Wr, 0.1, 0.005, true);
		MatrixXd Xcovar = randn((muint_t)N,(muint_t)Wr_covar);
		//y ~ w*X
		MatrixXd w = randn((muint_t)Wr,(muint_t)Wc);
		MatrixXd wcovar = randn((muint_t)Wr_covar,(muint_t)Wc_covar);

		MatrixXd A = MatrixXd::Identity((muint_t)Wc,(muint_t)P);
		MatrixXd A_inter = MatrixXd::Ones(1,(muint_t)P);
		MatrixXd A_covar = MatrixXd::Ones((muint_t)Wc_covar,(muint_t)P);
		MatrixXd wnoise1 = randn((muint_t)Xr1.cols(),(muint_t)Xc1.rows());
		MatrixXd wnoise2 = randn((muint_t)Xr2.cols(),(muint_t)Xc2.rows());
		std::cout<<"n1.sum()= "<<wnoise1.sum()<<"   n2.sum()= "<<wnoise2.sum()<<"\n";
		MatrixXd noise1 = Xr1*1.0*eps1*wnoise1*Xc1;
		MatrixXd noise2 = Xr2*1.0*eps2*wnoise2*Xc2;
		MatrixXd Y = X*w*A + Xcovar*wcovar*A_covar + noise1 + noise2;
		//SNPS: all random except for one true causal guy

		MatrixXd S = BaldingNichols((muint_t)N,(Nsnp0+1)*Wr, 0.1, 0.005, true);
		//S.block(0,0,N,Nsnp0) = randn((muint_t)N,Nsnp0*Wr);
		S.block(0,Nsnp0*Wr,N,Wr) = X;
		//std::cout<<"SNPS: "<< S<<std::endl;
		//2. construction of GP object

		MatrixXdVec Acov=MatrixXdVec();
		Acov.push_back(A_covar);
		MatrixXdVec Xcov=MatrixXdVec();
		Xcov.push_back(Xcovar);

		std::cout << A_covar.rows() << "," << A_covar.cols() << "," << Xcovar.rows() << "," << Xcovar.cols() << "\n";

		CKroneckerLMM lmm = CKroneckerLMM();
		lmm.setK1c(Kc1);
		lmm.setK1r(Kr1);
		lmm.setK2c(Kc2);
		lmm.setK2r(Kr2);
		lmm.setSNPs(S);
		lmm.setSNPcoldesign0_inter(A_inter);
		lmm.setSNPcoldesign(A);
		lmm.setCovariates(Xcov,Acov);
		lmm.setPheno(Y);
		lmm.setNumIntervalsAlt(0);
		lmm.setNumIntervals0_inter(0);
		lmm.setNumIntervals0(100);
		lmm.process();
		MatrixXd pv = MatrixXd();
		lmm.agetPv(&pv);
		mfloat_t pvmean = pv.mean();
		
		
		//std::cout << "pv:\n" << pv << "\n";
		std::cout << "mean(pv)="<<pvmean<<"  max(pv)="<< pv.maxCoeff() << "  min(pv)="<<pv.minCoeff()<<" cols:"<<pv.cols()<<" rows:"<<pv.rows()<<"\n";
		std::cout << "done.\n" ;
		
		}
		catch(CLimixException& e) {
			cout <<"Exception : "<< e.what() << endl;
		}

}

#endif
