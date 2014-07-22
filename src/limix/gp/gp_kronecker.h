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

#ifndef GP_KRONECKER_H_
#define GP_KRONECKER_H_

#include "gp_base.h"

namespace limix {


//forward definition:
class CGPkronecker;
class CGPKroneckerCache;


class CGPKroneckerCache : public CParamObject
{
	friend class CGPKronecker;
protected:
	MatrixXd YrotCache;
	MatrixXd SiCache;
	MatrixXd YSiCache;
	MatrixXd KinvYCache;
	mfloat_t KnoiseCache;

	CGPkronecker* gp;
	bool YrotCacheNull,SiCacheNull,YSiCacheNull,KinvYCacheNull,KnoiseCacheNull;
	//sync states
	Pbool syncLik,syncCovar_r,syncCovar_c,syncData;
	//validate & clear cache
	void validateCache();
public:
	PCovarianceFunctionCacheOld covar_r;
	PCovarianceFunctionCacheOld covar_c;

	CGPKroneckerCache(CGPkronecker* gp);
	virtual ~CGPKroneckerCache()
	{};
	MatrixXd& rgetYrot();
	MatrixXd& rgetSi();
	MatrixXd& rgetYSi();
	MatrixXd& rgetKinvY();

	void agetSi(MatrixXd* out)
	{
		(*out) = rgetSi();
	}
	void agetYSi(MatrixXd* out)
	{
		(*out) = rgetYSi();
	}
	void agetYrot(MatrixXd* out)
	{
		(*out) = rgetYrot();
	}
};
typedef sptr<CGPKroneckerCache> PGPKroneckerCache;


class CGPkronecker: public CGPbase {
	friend class CGPKroneckerCache;
	virtual void updateParams() ;

protected:
	//row and column covariance functions:
	PCovarianceFunction covar_r;
	PCovarianceFunction covar_c;


	//cache:
	PGPKroneckerCache cache;
	VectorXi gplvmDimensions_r;  //gplvm dimensions
	VectorXi gplvmDimensions_c;  //gplvm dimension

	mfloat_t _gradLogDet(MatrixXd& dK,bool columns);
	mfloat_t _gradQuadrForm(MatrixXd& dK,bool columns);
	void _gradQuadrFormX(VectorXd* rv,MatrixXd& dK,bool columns);
	void _gradLogDetX(VectorXd* out, MatrixXd& dK,bool columns);

public:
	CGPkronecker(PCovarianceFunction covar_r, PCovarianceFunction covar_c, PLikelihood lik=PLikelihood(),PDataTerm mean=PDataTerm());
	virtual ~CGPkronecker();

	void setX_r(const CovarInput& X) ;
	void setX_c(const CovarInput& X) ;
	void setY(const MatrixXd& Y);
	void setCovar_r(PCovarianceFunction covar);
	void setCovar_c(PCovarianceFunction covar);


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

	PLikNormalSVD getLik()
	{
		PLikNormalSVD RV = static_pointer_cast<CLikNormalSVD> (this->lik);
		return RV;
	}


	//predictions:
	virtual void apredictMean(MatrixXd* out, const MatrixXd& Xstar_r,const MatrixXd& Xstar_c) ;
	virtual void apredictVar(MatrixXd* out, const MatrixXd& Xstar_r,const MatrixXd& Xstar_c) ;


	CGPHyperParams LMLgrad() ;
	virtual void aLMLgrad_covar(VectorXd* out,bool columns) ;
	virtual void aLMLgrad_covar_r(VectorXd* out) ;
	virtual void aLMLgrad_covar_c(VectorXd* out) ;
	virtual void aLMLgrad_lik(VectorXd* out) ;
	virtual void aLMLgrad_X_r(MatrixXd* out) ;
	virtual void aLMLgrad_X_c(MatrixXd* out) ;
	virtual void aLMLgrad_dataTerm(MatrixXd* out) ;
    PGPKroneckerCache getCache();
    PCovarianceFunction  getCovarC() const;
    PCovarianceFunction  getCovarR() const;
    VectorXi getGplvmDimensionsC() const;
    VectorXi getGplvmDimensionsR() const;
    void setGplvmDimensionsC(VectorXi gplvmDimensionsC);
    void setGplvmDimensionsR(VectorXi gplvmDimensionsR);
};
typedef sptr<CGPkronecker> PGPkronecker;


} /* namespace limix */
#endif /* GP_KRONECKER_H_ */
