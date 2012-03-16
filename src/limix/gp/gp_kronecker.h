/*
 * gp_kronecker.h
 *
 *  Created on: Jan 2, 2012
 *      Author: stegle
 */

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
	PCovarianceFunctionCache covar_r;
	PCovarianceFunctionCache covar_c;

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

#if (defined(SWIG) && !defined(SWIG_FILE_WITH_INIT))
%ignore CGPkronecker::predictMean;
%ignore CGPkronecker::predictVar;

%rename(predictMean) CGPkronecker::apredictMean;
%rename(predictVar) CGPkronecker::apredictVar;
#endif

class CGPkronecker: public CGPbase {
	friend class CGPKroneckerCache;
	virtual void updateParams() throw (CGPMixException);

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

	void setX_r(const CovarInput& X) throw (CGPMixException);
	void setX_c(const CovarInput& X) throw (CGPMixException);
	void setY(const MatrixXd& Y);
	void setCovar_r(PCovarianceFunction covar);
	void setCovar_c(PCovarianceFunction covar);


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

	PLikNormalSVD getLik()
	{
		PLikNormalSVD RV = std::tr1::static_pointer_cast<CLikNormalSVD> (this->lik);
		return RV;
	}


	//predictions:
	virtual void apredictMean(MatrixXd* out, const MatrixXd& Xstar_r,const MatrixXd& Xstar_c) throw (CGPMixException);
	virtual void apredictVar(MatrixXd* out, const MatrixXd& Xstar_r,const MatrixXd& Xstar_c) throw (CGPMixException);


	CGPHyperParams LMLgrad() throw (CGPMixException);
	virtual void aLMLgrad_covar(VectorXd* out,bool columns) throw (CGPMixException);
	virtual void aLMLgrad_covar_r(VectorXd* out) throw (CGPMixException);
	virtual void aLMLgrad_covar_c(VectorXd* out) throw (CGPMixException);
	virtual void aLMLgrad_lik(VectorXd* out) throw (CGPMixException);
	virtual void aLMLgrad_X_r(MatrixXd* out) throw (CGPMixException);
	virtual void aLMLgrad_X_c(MatrixXd* out) throw (CGPMixException);
	virtual void aLMLgrad_dataTerm(MatrixXd* out) throw (CGPMixException);
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
