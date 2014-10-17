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

#ifndef GP_KronSum_H_
#define GP_KronSum_H_

#include "gp_base.h"

//debug
#include "limix/utils/matrix_helper.h"

namespace limix {

//forward definition:
class CGPkronSum;
class CGPkronSumCache;

/**
Stores all the intermediate computations for the two covariance Kronecker model

- K        = C \kron R + Sigma \kron Omega
- Sigma    = Usigma diag(Ssigma) Usigma.T
- Omega    = Uomega diag(Somega) Uomega.T
- Cstar    = diag(Ssigma)^(-0.5) Usigma.T C Usigma.T diag(Ssigma)^(-0.5)
- Rstar    = diag(Somega)^(-0.5) Uomega.T R Uomega.T diag(Somega)^(-0.5)
- Cstar    = Ucstar diag(Scstar) Ucstar.T
- Rstar    = Urstar diag(Srstar) Urstar.T
- Lambdac  = Ucstar diag(Ssigma^(-0.5)) Usigma
- Lambdar  = Uomega diag(Somega^(-0.5)) Uomega
- D        = vec^(-1)((Scstar \kron diag(Srstar) + I)^(-1))
- YrotPart = vec^(-1)((I \kron Lambdar) vec(Y) )
- Yrot     = vec^(-1)((Lambdac \kron I) vec(YrotPart) )
- Ytilde   = D.array() * Yrot.array()
*/
class CGPkronSumCache : public CParamObject
{
	friend class CGPkronSum;
protected:
	MatrixXd SsigmaCache;
	MatrixXd ScstarCache;
	MatrixXd UcstarCache;
	MatrixXd LambdacCache;
	MatrixXd SomegaCache;
	MatrixXd SrstarCache;
	MatrixXd UrstarCache;
	MatrixXd LambdarCache;
	MatrixXd DCache;
	MatrixXd YrotPartCache;
	MatrixXd YrotCache;
	MatrixXd YtildeCache;

	CGPkronSum* gp;
	bool SVDcstarCacheNull,SVDrstarCacheNull;
	bool LambdarCacheNull,LambdacCacheNull,DCacheNull;
	bool YrotPartCacheNull,YrotCacheNull,YtildeCacheNull;
	//sync states
	Pbool syncLik,syncCovarc1,syncCovarc2,syncCovarr1,syncCovarr2,syncData;
	//validate & clear cache
	void validateCache();
	void updateSVDcstar();
	void updateSVDrstar();
public:
	PCovarianceFunctionCacheOld covarc1;
	PCovarianceFunctionCacheOld covarc2;
	PCovarianceFunctionCacheOld covarr1;
	PCovarianceFunctionCacheOld covarr2;

	CGPkronSumCache(CGPkronSum* gp);
	virtual ~CGPkronSumCache()	{};

	MatrixXd& rgetSsigma();
	MatrixXd& rgetScstar();
	MatrixXd& rgetUcstar();
	MatrixXd& rgetLambdac();
	MatrixXd& rgetSomega();
	MatrixXd& rgetSrstar();
	MatrixXd& rgetUrstar();
	MatrixXd& rgetLambdar();
	MatrixXd& rgetD();
	MatrixXd& rgetYrotPart();
	MatrixXd& rgetYrot();
	MatrixXd& rgetYtilde();

	void argetSsigma(MatrixXd* out)
	{
		(*out) = rgetSsigma();
	}
	void argetScstar(MatrixXd* out)
	{
		(*out) = rgetScstar();
	}
	void argetUcstar(MatrixXd* out)
	{
		(*out) = rgetUcstar();
	}
	void argetLambdac(MatrixXd* out)
	{
		(*out) = rgetLambdac();
	}
	void argetSomega(MatrixXd* out)
	{
		(*out) = rgetSomega();
	}
	void argetSrstar(MatrixXd* out)
	{
		(*out) = rgetSrstar();
	}
	void argetUrstar(MatrixXd* out)
	{
		(*out) = rgetUrstar();
	}
	void argetLambdar(MatrixXd* out)
	{
		(*out) = rgetLambdar();
	}
	void argetD(MatrixXd* out)
	{
		(*out) = rgetD();
	}
	void argetYrotPart(MatrixXd* out)
	{
		(*out) = rgetYrotPart();
	}
	void argetYrot(MatrixXd* out)
	{
		(*out) = rgetYrot();
	}
	void argetYtilde(MatrixXd* out)
	{
		(*out) = rgetYtilde();
	}

};
typedef sptr<CGPkronSumCache> PGPkronSumCache;


class CGPkronSum: public CGPbase {
	friend class CGPkronSumCache;
	virtual void updateParams() ;

protected:
	//row and column covariance functions:
	PCovarianceFunction covarc1;
	PCovarianceFunction covarc2;
	PCovarianceFunction covarr1;
	PCovarianceFunction covarr2;
	//cache:
	PGPkronSumCache cache;

	//dimensions
	muint_t N;
	muint_t P;

	//penalization
	mfloat_t lambda_g;
	mfloat_t lambda_n;

	//debug bool
	bool debug;

	//running times
	mfloat_t rtSVDcols;
	mfloat_t rtSVDrows;
	mfloat_t rtLambdac;
	mfloat_t rtLambdar;
	mfloat_t rtD;
	mfloat_t rtYrotPart;
	mfloat_t rtYrot;
	mfloat_t rtYtilde;
	mfloat_t rtCC1part1a;
	mfloat_t rtCC1part1b;
	mfloat_t rtCC1part1c;
	mfloat_t rtCC1part1d;
	mfloat_t rtCC1part1e;
	mfloat_t rtCC1part1f;
	mfloat_t rtCC1part2;
	mfloat_t rtCC2part1;
	mfloat_t rtCC2part2;
	mfloat_t rtCR1part1a;
	mfloat_t rtCR1part1b;
	mfloat_t rtCR1part2;
	mfloat_t rtCR2part1a;
	mfloat_t rtCR2part1b;
	mfloat_t rtCR2part2;
	mfloat_t rtLMLgradCovar;
	mfloat_t rtLMLgradDataTerm;
	mfloat_t is_it;
	mfloat_t rtGrad;
	mfloat_t rtLML1a;
	mfloat_t rtLML1b;
	mfloat_t rtLML1c;
	mfloat_t rtLML1d;
	mfloat_t rtLML1e;
	mfloat_t rtLML2;
	mfloat_t rtLML3;
	mfloat_t rtLML4;


public:
	CGPkronSum(const MatrixXd& Y,PCovarianceFunction covarr1, PCovarianceFunction covarc1,PCovarianceFunction covarr2, PCovarianceFunction covarc2, PLikelihood lik, PDataTerm dataTerm);
	virtual ~CGPkronSum();

	//set penalization constant
	virtual void setLambda(mfloat_t lambda) {
		this->setLambdaG(lambda);
		this->setLambdaN(lambda);
	}
	virtual void setLambdaG(mfloat_t lambda) {this->lambda_g=lambda;};
	virtual void setLambdaN(mfloat_t lambda) {this->lambda_n=lambda;};

	//getter for parameter bounds and hyperparam Mask
	virtual CGPHyperParams getParamBounds(bool upper) const;
	virtual CGPHyperParams getParamMask() const;

	//get Covariances
	PCovarianceFunction getCovarr1() {return covarr1;};
	PCovarianceFunction getCovarr2() {return covarr2;};
	PCovarianceFunction getCovarc1() {return covarc1;};
	PCovarianceFunction getCovarc2() {return covarc2;};
	//get from cache
	virtual void agetKEffInvYCache(MatrixXd* out) ;

	// LML
	mfloat_t LML() ;

	// Gradient
	CGPHyperParams LMLgrad() ;
	virtual void aLMLgrad_covarc1(VectorXd* out) ;
	virtual void aLMLgrad_covarc2(VectorXd* out) ;
	virtual void aLMLgrad_covarr1(VectorXd* out) ;
	virtual void aLMLgrad_covarr2(VectorXd* out) ;
	virtual void aLMLgrad_dataTerm(MatrixXd* out) ;

	// Hessian
	virtual void aLMLhess_c1c1(MatrixXd* out) ;
	//virtual void aLMLhess_c1r1(MatrixXd* out) ;
	//virtual void aLMLhess_c1c2(MatrixXd* out) ;
	//virtual void aLMLhess_c1r2(MatrixXd* out) ;
	//virtual void aLMLhess_r1r1(MatrixXd* out) ;
	//virtual void aLMLhess_r1c2(MatrixXd* out) ;
	//virtual void aLMLhess_r1r2(MatrixXd* out) ;
	//virtual void aLMLhess_c2c2(MatrixXd* out) ;
	//virtual void aLMLhess_c2r2(MatrixXd* out) ;
	//virtual void aLMLhess_r2r2(MatrixXd* out) ;


	/* DEBUGGING */

	//cache stuff
	void agetSc(MatrixXd *out) {(*out)=cache->covarc1->rgetSK();}
	void agetUc(MatrixXd *out) {(*out)=cache->covarc1->rgetUK();}
	void agetSr(MatrixXd *out) {(*out)=cache->covarr1->rgetSK();}
	void agetUr(MatrixXd *out) {(*out)=cache->covarr1->rgetUK();}
	void agetSsigma(MatrixXd *out) {(*out)=cache->covarc2->rgetSK();}
	void agetUsigma(MatrixXd *out) {(*out)=cache->covarc2->rgetUK();}
	void agetSomega(MatrixXd *out) {(*out)=cache->covarr2->rgetSK();}
	void agetUomega(MatrixXd *out) {(*out)=cache->covarr2->rgetUK();}
	void agetScstar(MatrixXd *out) {(*out)=cache->rgetScstar();}
	void agetUcstar(MatrixXd *out) {(*out)=cache->rgetUcstar();}
	void agetSrstar(MatrixXd *out) {(*out)=cache->rgetSrstar();}
	void agetUrstar(MatrixXd *out) {(*out)=cache->rgetUrstar();}
	void agetLambdac(MatrixXd *out) {(*out)=cache->rgetLambdac();}
	void agetLambdar(MatrixXd *out) {(*out)=cache->rgetLambdar();}
	void agetYrotPart(MatrixXd *out) {(*out)=cache->rgetYrotPart();}
	void agetYrot(MatrixXd *out) {(*out)=cache->rgetYrot();}
	void agetCstar(MatrixXd *out)
	{
		MatrixXd USisqrt;
		aUS2alpha(USisqrt,cache->covarc2->rgetUK(),cache->covarc2->rgetSK(),-0.5);
		(*out) = USisqrt.transpose()*cache->covarc1->rgetK()*USisqrt;
	}

	//dimensions
	muint_t getN() 	{return N;}
	muint_t getP() 	{return P;}

	//debug mode
	void setDebugMode(bool debug) {this->debug=debug;}

	//running times
	mfloat_t getRtSVDcols() 	{return rtSVDcols;}
	mfloat_t getRtSVDrows() 	{return rtSVDrows;}
	mfloat_t getRtLambdac() 	{return rtLambdac;}
	mfloat_t getRtLambdar() 	{return rtLambdar;}
	mfloat_t getRtD() 			{return rtD;}
	mfloat_t getRtYrotPart() 	{return rtYrotPart;}
	mfloat_t getRtYrot() 		{return rtYrot;}
	mfloat_t getRtYtilde() 		{return rtYtilde;}
	mfloat_t getRtCC1part1a() 	{return rtCC1part1a;}
	mfloat_t getRtCC1part1b() 	{return rtCC1part1b;}
	mfloat_t getRtCC1part1c() 	{return rtCC1part1c;}
	mfloat_t getRtCC1part1d() 	{return rtCC1part1d;}
	mfloat_t getRtCC1part1e() 	{return rtCC1part1e;}
	mfloat_t getRtCC1part1f() 	{return rtCC1part1f;}
	mfloat_t getRtCC1part2() 	{return rtCC1part2;}
	mfloat_t getRtCC2part1() 	{return rtCC2part1;}
	mfloat_t getRtCC2part2() 	{return rtCC2part2;}
	mfloat_t getRtCR1part1a() 	{return rtCR1part1a;}
	mfloat_t getRtCR1part1b() 	{return rtCR1part1b;}
	mfloat_t getRtCR1part2() 	{return rtCR1part2;}
	mfloat_t getRtCR2part1a() 	{return rtCR2part1a;}
	mfloat_t getRtCR2part1b() 	{return rtCR2part1b;}
	mfloat_t getRtCR2part2() 	{return rtCR2part2;}
	mfloat_t getRtLMLgradCovar() 	{return rtLMLgradCovar;}
	mfloat_t getRtLMLgradDataTerm() 	{return rtLMLgradDataTerm;}
	mfloat_t getIs_it() 	{return is_it;}
	mfloat_t getRtGrad() 	{return rtGrad;}
	mfloat_t getRtLML1a() 	{return rtLML1a;}
	mfloat_t getRtLML1b() 	{return rtLML1b;}
	mfloat_t getRtLML1c() 	{return rtLML1c;}
	mfloat_t getRtLML1d() 	{return rtLML1d;}
	mfloat_t getRtLML1e() 	{return rtLML1e;}
	mfloat_t getRtLML2() 	{return rtLML2;}
	mfloat_t getRtLML3() 	{return rtLML3;}
	mfloat_t getRtLML4() 	{return rtLML4;}

};
typedef sptr<CGPkronSum> PGPkronSum;


} /* namespace limix */
#endif /* GP_KronSum_H_ */

