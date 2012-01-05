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

//forward definition:
class CGPkronecker;
class CGPKroneckerCache;

//caching module for SVD
class GPSVDCache : public CGPCholCache
{
	friend class CGPKroneckerCache;
protected:
	MatrixXd UK;
	MatrixXd SK;
	ACovarianceFunction& covar;
public:
	GPSVDCache(CGPbase& gp, ACovarianceFunction& covar) : CGPCholCache(gp), covar(covar)
	{};
	virtual ~GPSVDCache()
	{};
	virtual void clearCache();
	virtual bool isInSync() const;

	MatrixXd* getUK();
	MatrixXd* getSK();
};

class CGPKroneckerCache
{
protected:
	MatrixXd yrot;
	MatrixXd Si;
	MatrixXd YSi;
	MatrixXd Knoise;
	CGPkronecker& gp;
public:
	CGPKroneckerCache(CGPkronecker& gp) :  gp(gp)
	{};
	virtual ~CGPKroneckerCache()
	{};
	virtual void clearCache();
	virtual bool isInSync() const;



};

class CGPkronecker: public CGPbase {
	friend class CGPKroneckerCache;
protected:
	//row and column covariance functions:
	ACovarianceFunction& covar_r;
	ACovarianceFunction& covar_c;
	//cache:
	GPSVDCache cache_r;
	GPSVDCache cache_c;


	VectorXi gplvmDimensions_r;  //gplvm dimensions
	VectorXi gplvmDimensions_c;  //gplvm dimensions
	virtual void updateParams() throw (CGPMixException);
public:
	CGPkronecker(ADataTerm& mean, ACovarianceFunction& covar_r, ACovarianceFunction& covar_c, ALikelihood& lik);
	virtual ~CGPkronecker();
};

} /* namespace gpmix */
#endif /* GP_KRONECKER_H_ */
