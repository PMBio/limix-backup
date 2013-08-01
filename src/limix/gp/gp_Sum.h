/*
 * gp_kronecker.h
 *
 *  Created on: Jul 29, 2013
 *      Author: casale
 */

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
	PCovarianceFunctionCache covar1;
	PCovarianceFunctionCache covar2;

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



#if (defined(SWIG) && !defined(SWIG_FILE_WITH_INIT))
%ignore CGPSum::predictMean;
%ignore CGPSum::predictVar;

%rename(predictMean) CGPSum::apredictMean;
%rename(predictVar) CGPSum::apredictVar;
#endif

class CGPSum: public CGPbase {
	friend class CGPSumCache;
	virtual void updateParams() throw (CGPMixException);

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

	void setX1(const CovarInput& X) throw (CGPMixException);
	void setX2(const CovarInput& X) throw (CGPMixException);
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


	mfloat_t LML() throw (CGPMixException);
	virtual mfloat_t LML(const CGPHyperParams& params) throw (CGPMixException)
	{
		return CGPbase::LML(params);
	}
	//same for concatenated list of parameters
	virtual mfloat_t LML(const VectorXd& params) throw (CGPMixException)
	{
		return CGPbase::LML(params);
	}

	// Gradient Stuff
	CGPHyperParams LMLgrad() throw (CGPMixException);
	virtual void aLMLgrad_covar(VectorXd* out, bool cov1) throw (CGPMixException);
	virtual void aLMLgrad_covar1(VectorXd* out) throw (CGPMixException);
	virtual void aLMLgrad_covar2(VectorXd* out) throw (CGPMixException);
	virtual void aLMLgrad_dataTerm(MatrixXd* out) throw (CGPMixException);

	/*
	//predictions:
	virtual void apredictMean(MatrixXd* out, const MatrixXd& Xstar_r,const MatrixXd& Xstar_c) throw (CGPMixException);
	virtual void apredictVar(MatrixXd* out, const MatrixXd& Xstar_r,const MatrixXd& Xstar_c) throw (CGPMixException);

	//Gradient Stuff
	virtual void aLMLgrad_X_r(MatrixXd* out) throw (CGPMixException);
	virtual void aLMLgrad_X_c(MatrixXd* out) throw (CGPMixException);
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
