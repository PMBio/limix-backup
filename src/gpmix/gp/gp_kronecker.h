/*
 * gp_kronecker.h
 *
 *  Created on: Jan 2, 2012
 *      Author: stegle
 */

#ifndef GP_KRONECKER_H_
#define GP_KRONECKER_H_

#include "gp_base.h"

namespace gpmix {


//math utils
void akronravel(MatrixXd* out, const MatrixXd& A, const MatrixXd& B, const MatrixXd& X);
MatrixXd kronravel(const MatrixXd& A, const MatrixXd& B, const MatrixXd& X);
void akrondiag(MatrixXd* out, const VectorXd& v1, const VectorXd& v2);
MatrixXd krondiag(const VectorXd& v1, const VectorXd& v2);


//forward definition:
class CGPkronecker;
class CGPKroneckerCache;

//caching module for SVD
class CGPSVDCache : public CGPCholCache
{
	friend class CGPKroneckerCache;
protected:
	MatrixXd UK;
	VectorXd SK;
	ACovarianceFunction* covar;
public:
	CGPSVDCache(CGPbase* gp, ACovarianceFunction* covar) : CGPCholCache(gp,covar), covar(covar)
	{};
	virtual ~CGPSVDCache()
	{};
	virtual void clearCache();
	virtual bool isInSync() const;

	MatrixXd& getUK();
	VectorXd& getSK();
    ACovarianceFunction& getCovar();
    void agetUK(MatrixXd* out)
    {
    	(*out) = getUK();
    }
    void agetSK(VectorXd* out)
    {
    	(*out) = getSK();
    }

};

class CGPKroneckerCache
{
	friend class CGPKronecker;
protected:
	MatrixXd Yrot;
	MatrixXd Si;
	MatrixXd YSi;
	mfloat_t Knoise;
	CGPbase* gp;

public:
	CGPSVDCache cache_r;
	CGPSVDCache cache_c;

	CGPKroneckerCache(CGPbase* gp,ACovarianceFunction* covar_r,ACovarianceFunction* covar_c );
	virtual ~CGPKroneckerCache()
	{};
	virtual void clearCache();
	virtual bool isInSync() const;
	MatrixXd& getYrot();
	MatrixXd& getSi();
	MatrixXd& getYSi();
	mfloat_t getKnoise();

	void agetSi(MatrixXd* out)
	{
		(*out) = getSi();
	}
	void agetYSi(MatrixXd* out)
	{
		(*out) = getYSi();
	}
	void agetYrot(MatrixXd* out)
	{
		(*out) = getYrot();
	}
};

class CGPkronecker: public CGPbase {
	friend class CGPKroneckerCache;
	virtual void updateParams() throw (CGPMixException);

protected:
	//row and column covariance functions:
	ACovarianceFunction& covar_r;
	ACovarianceFunction& covar_c;
	//cache:
	CGPKroneckerCache cache;

	VectorXi gplvmDimensions_r;  //gplvm dimensions
	VectorXi gplvmDimensions_c;  //gplvm dimension

	mfloat_t _gradLogDet(MatrixXd& dK,bool columns);
	mfloat_t _gradQuadrForm(MatrixXd& dK,bool columns);
	void _gradQuadrFormX(VectorXd* rv,MatrixXd& dK,bool columns);
	void _gradLogDetX(VectorXd* out, MatrixXd& dK,bool columns);

public:
	CGPkronecker(ADataTerm& mean, ACovarianceFunction& covar_r, ACovarianceFunction& covar_c, ALikelihood& lik);
	virtual ~CGPkronecker();

	void setX_r(const CovarInput& X) throw (CGPMixException);
	void setX_c(const CovarInput& X) throw (CGPMixException);
	void setY(const MatrixXd& Y);

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


	CGPHyperParams LMLgrad() throw (CGPMixException);
	virtual void aLMLgrad_covar(VectorXd* out,bool columns) throw (CGPMixException);
	virtual void aLMLgrad_covar_r(VectorXd* out) throw (CGPMixException);
	virtual void aLMLgrad_covar_c(VectorXd* out) throw (CGPMixException);
	virtual void aLMLgrad_lik(VectorXd* out) throw (CGPMixException);
	virtual void aLMLgrad_X_r(MatrixXd* out) throw (CGPMixException);
	virtual void aLMLgrad_X_c(MatrixXd* out) throw (CGPMixException);
    CGPKroneckerCache* getCache();
    ACovarianceFunction & getCovarC() const;
    ACovarianceFunction & getCovarR() const;
    VectorXi getGplvmDimensionsC() const;
    VectorXi getGplvmDimensionsR() const;
    void setGplvmDimensionsC(VectorXi gplvmDimensionsC);
    void setGplvmDimensionsR(VectorXi gplvmDimensionsR);
};

} /* namespace gpmix */
#endif /* GP_KRONECKER_H_ */
