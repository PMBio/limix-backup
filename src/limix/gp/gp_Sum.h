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

#ifndef GP_Sum_H_
#define GP_Sum_H_

#include "gp_base.h"

namespace limix {

//forward definition:
class CGPSum;
class CGPSumCache;

class CGPSumCache : public CParamObject
{
	friend class CGPSum;
protected:
	MatrixXd ScstarCache;
	MatrixXd UcstarCache;
	MatrixXd LambdaCache;
	MatrixXd YrotCache;
	MatrixXd FrotCache;

	CGPSum* gp;
	bool SVDcstarCacheNull,LambdaCacheNull,YrotCacheNull;
	//sync states
	Pbool syncLik,syncCovar1,syncCovar2,syncData;
	//validate & clear cache
	void validateCache();
	void updateSVDcstar();
public:
	PCovarianceFunctionCacheOld covar1;
	PCovarianceFunctionCacheOld covar2;

	CGPSumCache(CGPSum* gp);
	virtual ~CGPSumCache()	{};

	MatrixXd& rgetScstar();
	MatrixXd& rgetUcstar();
	MatrixXd& rgetLambda();
	MatrixXd& rgetYrot();

	void argetScstar(MatrixXd* out)
	{
		(*out) = rgetScstar();
	}
	void argetUcstar(MatrixXd* out)
	{
		(*out) = rgetUcstar();
	}
	void argetLambda(MatrixXd* out)
	{
		(*out) = rgetLambda();
	}
	void argetYrot(MatrixXd* out)
	{
		(*out) = rgetYrot();
	}

};
typedef sptr<CGPSumCache> PGPSumCache;


class CGPSum: public CGPbase {
	friend class CGPSumCache;
	virtual void updateParams() ;

protected:
	//row and column covariance functions:
	PCovarianceFunction covar1;
	PCovarianceFunction covar2;

	//cache:
	PGPSumCache cache;
	VectorXi gplvmDimensions1;  //gplvm dimensions
	VectorXi gplvmDimensions2;  //gplvm dimension

public:
	CGPSum(const MatrixXd& Y,PCovarianceFunction covar1, PCovarianceFunction covar2, PLikelihood lik, PDataTerm dataTerm);
	virtual ~CGPSum();

	void setX1(const CovarInput& X) ;
	void setX2(const CovarInput& X) ;
	void setY(const MatrixXd& Y);
	void setCovar1(PCovarianceFunction covar);
	void setCovar2(PCovarianceFunction covar);

	PCovarianceFunction getCovar1() {return this->covar1;};
	PCovarianceFunction getCovar2() {return this->covar2;};

	//cache stuff
	void agetScstar(MatrixXd *out) {(*out)=cache->rgetScstar();}
	void agetLambda(MatrixXd *out) {(*out)=cache->rgetLambda();}
	void agetYrot(MatrixXd *out) {(*out)=cache->rgetYrot();}
	void debugCache();
	PGPSumCache agetCache() {return this->cache;}


	mfloat_t LML() ;
	virtual mfloat_t LML(const CGPHyperParams& params) 
	{
		return CGPbase::LML(params);
	}
	//same for concatenated list of parameters
	virtual mfloat_t LML(const VectorXd& params) 
	{
		return CGPbase::LML(params);
	}

	// Gradient Stuff
	CGPHyperParams LMLgrad() ;
	virtual void aLMLgrad_covar(VectorXd* out, bool cov1) ;
	virtual void aLMLgrad_covar1(VectorXd* out) ;
	virtual void aLMLgrad_covar2(VectorXd* out) ;
	virtual void aLMLgrad_dataTerm(MatrixXd* out) ;

	/*
	//predictions:
	virtual void apredictMean(MatrixXd* out, const MatrixXd& Xstar_r,const MatrixXd& Xstar_c) ;
	virtual void apredictVar(MatrixXd* out, const MatrixXd& Xstar_r,const MatrixXd& Xstar_c) ;

	//Gradient Stuff
	virtual void aLMLgrad_X_r(MatrixXd* out) ;
	virtual void aLMLgrad_X_c(MatrixXd* out) ;
	mfloat_t _gradLogDet(MatrixXd& dK,bool columns);
	mfloat_t _gradQuadrForm(MatrixXd& dK,bool columns);
	void _gradQuadrFormX(VectorXd* rv,MatrixXd& dK,bool columns);
	void _gradLogDetX(VectorXd* out, MatrixXd& dK,bool columns);
    PGPKroneckerCache getCache();
    PCovarianceFunction  getCovarC() const;
    PCovarianceFunction  getCovarR() const;
    VectorXi getGplvmDimensionsC() const;
    VectorXi getGplvmDimensionsR() const;
    void setGplvmDimensionsC(VectorXi gplvmDimensionsC);
    void setGplvmDimensionsR(VectorXi gplvmDimensionsR);
    */
};
typedef sptr<CGPSum> PGPSum;


} /* namespace limix */
#endif /* GP_Sum_H_ */
