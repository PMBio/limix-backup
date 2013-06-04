/*
 * gp_base.h
 *
 *  Created on: Nov 11, 2011
 *      Author: stegle
 */

#ifndef GP_BASE_H_
#define GP_BASE_H_

//#include "limix/covar/covariance.h"
#include "limix/covar/freeform.h"
#include "limix/covar/combinators.h"
#include "limix/likelihood/likelihood.h"
//#include "limix/mean/ADataTerm.h"
#include "limix/mean/CLinearMean.h"
#include "limix/mean/CData.h"
#include "limix/types.h"

#include <string>
#include <map>
#include <vector>
#include <iostream>


namespace limix {

/* Forward declaratins of classes */
class CGPbase;
class CGPKroneckerCache;
typedef sptr<CGPbase> PGPbase;



#if (defined(SWIG) && !defined(SWIG_FILE_WITH_INIT))
%ignore CGPHyperParams::get;
%ignore CGPHyperParams::getParamArray;

%rename(get) CGPHyperParams::aget;
%rename(getParamArray) CGPHyperParams::agetParamArray;
//%template(StringVec) std::vector<std::string>;
//PYTHON:
#ifdef SWIGPYTHON
%rename(__getitem__) CGPHyperParams::aget;
%rename(__setitem__) CGPHyperParams::set;
%rename(__str__) CGPHyperParams::toString;
#endif
#endif
//typedef map<string,MatrixXd> CGPHyperParamsMap;
class CGPHyperParams : public std::map<std::string,MatrixXd>
{
	/*
	 * CGHyperParams:
	 * helper class to handle different types of paramters
	 * Map: string -> MatrixXd
	 * Parameters can be vectors or matrices (MatrixXd)
	 *
	 * if set using .set(), the current structure of the parameter array is destroyed.
	 * if set using .setArrayParams(), the current structure is enforced.
	 * Usage:
	 * - set the structure using repeated calls of .set(name,value)
	 * - once built, optimizers and CGPbase rely on setParamArray(), getParamArra() to convert the
	 *   readle representation of parameters to and from a vectorial one.
	 */
protected:
	MatrixXd filterMask(const MatrixXd& array,const MatrixXd& mask) const;
	void expandMask(MatrixXd& out,const MatrixXd& array,const MatrixXd& mask) const;
	//	MatrixXd filterMask(const MatrixXd& array,const MatrixXd& mask) const;

public:
	CGPHyperParams()
	{
	}
	//copy constructor
	CGPHyperParams(const CGPHyperParams &_param);

	//from a list of params
	~CGPHyperParams()
	{
	}

	void agetParamArray(VectorXd* out) const throw(CGPMixException);
	void setParamArray(const VectorXd& param) throw (CGPMixException);

	void agetParamArray(VectorXd* out,const CGPHyperParams& mask) const throw(CGPMixException);
	void setParamArray(const VectorXd& param,const CGPHyperParams& mask) throw (CGPMixException);

	muint_t getNumberParams() const;
	muint_t getNumberParams(const CGPHyperParams& mask) const;


	void set(const std::string& name, const MatrixXd& value);
	void aget(MatrixXd* out, const std::string& name);
	//get vector with existing names
	std::vector<std::string> getNames() const;
	//exists?
	bool exists(std::string name) const;

	std::string toString() const;
	//operator overloading
	friend std::ostream& operator <<(std::ostream &os,const CGPHyperParams &obj);

	//convenience functions for C++ access
	inline MatrixXd get(const std::string& name);
	inline VectorXd getParamArray() const;
	inline VectorXd getParamArray(const CGPHyperParams& mask) const throw (CGPMixException);
};
typedef sptr<CGPHyperParams> PGPHyperParams;


inline MatrixXd CGPHyperParams::get(const std::string& name)
{
	MatrixXd rv;
	aget(&rv,name);
	return rv;
}
inline VectorXd CGPHyperParams::getParamArray() const
{
	VectorXd rv;
	agetParamArray(&rv);
	return rv;
}

inline VectorXd CGPHyperParams::getParamArray(const CGPHyperParams& mask) const throw (CGPMixException)
{
	VectorXd rv;
	agetParamArray(&rv,mask);
	return rv;
}



//Gaussian process caching class for Cholesky-basd inference
class CGPCholCache : public CParamObject
{
protected:
	MatrixXd KEffCache;
	MatrixXdChol KEffCholCache;
	MatrixXd KEffInvCache;
	MatrixXd KEffInvYCache;
	MatrixXd DKinv_KEffinvYYKEffinvCache;
	MatrixXd YeffectiveCache;
	bool KEffCacheNull,KEffCholNull,KEffInvCacheNull,KEffInvYCacheNull,DKinv_KEffInvYYKEffInvCacheNull,YeffectiveCacheNull;
	//lik, covar and data term sync state
	Pbool syncLik,syncCovar,syncData;
	//TODO change this to shared pointer
	CGPbase* gp;
	PCovarianceFunctionCache covar;
	//validate & clear cache
	void validateCache();
public:
	CGPCholCache(CGPbase* gp);
	virtual ~CGPCholCache()
	{};
	PCovarianceFunctionCache getCovar()
	{
		return covar;
	}
	void setCovar(PCovarianceFunction covar);


	virtual MatrixXd& rgetKEff();
	virtual MatrixXdChol& rgetKEffChol();
	virtual MatrixXd& rgetKEffInv();
	virtual MatrixXd& rgetYeffective();
	virtual MatrixXd& rgetKEffInvY();
	virtual MatrixXd& getDKEffInv_KEffInvYYKinv();
};
typedef sptr<CGPCholCache> PGPCholCache;

#if (defined(SWIG) && !defined(SWIG_FILE_WITH_INIT))
%ignore CGPbase::getX;
%ignore CGPbase::getY;
%ignore CGPbase::LMLgrad_covar;
%ignore CGPbase::LMLgrad_lik;
%ignore CGPbase::getParamArray;
%ignore CGPbase::predictMean;
%ignore CGPbase::predictVar;

%rename(getParamArray) CGPbase::agetParamArray;
%rename(getX) CGPbase::agetX;
%rename(getY) CGPbase::agetY;
%rename(LMLgrad_covar) CGPbase::aLMLgrad_covar;
%rename(LMLgrad_lik) CGPbase::aLMLgrad_lik;
%rename(LMLhess) CGPbase::aLMLhess;
%rename(LMLhess_covar) CGPbase::aLMLhess_covar;
%rename(LMLhess_lik) CGPbase::aLMLhess_lik;
%rename(LMLhess_covarlik) CGPbase::aLMLhess_covarlik;
%rename(getCov_laplace) CGPbase::agetCov_laplace;
%rename(getStd_laplace) CGPbase::agetStd_laplace;
%rename(predictMean) CGPbase::apredictMean;
%rename(predictVar) CGPbase::apredictVar;
//
//%sptr(gpmix::CGPbase)
#endif



class CGPbase : public enable_shared_from_this<CGPbase> {
	friend class CGPCholCache;
	friend class CGPKroneckerCache;
protected:

	//cached GP-parameters:
	PGPCholCache cache;
	CGPHyperParams params;

	//smart pointers for data Term,covar,lik
	PDataTerm dataTerm;       	//Mean function
	PCovarianceFunction covar;	//Covariance function
	PLikelihood lik;          	//likelihood model

	VectorXi gplvmDimensions;  //gplvm dimensions

	virtual void updateParams() throw (CGPMixException);
	void updateX(ACovarianceFunction& covar,const VectorXi& gplvmDimensions,const MatrixXd& X) throw (CGPMixException);

public:
	CGPbase(PCovarianceFunction covar, PLikelihood lik=PLikelihood(),PDataTerm data=PDataTerm());
	virtual ~CGPbase();

	//TODO: add interface that is suitable for optimizer
	// virtual double LML(double* params);
	// virtual void LML(double* params, double* gradients);
	virtual void set_data(MatrixXd& Y);

	virtual void setCovar(PCovarianceFunction covar);
	virtual void setLik(PLikelihood lik);
	virtual void setDataTerm(PDataTerm data);



	//getter and setter for Parameters:
	virtual void setParams(const CGPHyperParams& hyperparams) throw(CGPMixException);
	virtual void setParams(const CGPHyperParams& hyperparams,const CGPHyperParams& mask) throw(CGPMixException);
	virtual CGPHyperParams getParams() const;
	virtual void setParamArray(const VectorXd& hyperparams) throw (CGPMixException);
	virtual void setParamArray(const VectorXd& param,const CGPHyperParams& mask) throw (CGPMixException);
	virtual void agetParamArray(VectorXd* out) const;

	//getter for parameter bounds and hyperparam Mask
	virtual CGPHyperParams getParamBounds(bool upper) const;
	virtual CGPHyperParams getParamMask() const;

	void agetY(MatrixXd* out);
	void setY(const MatrixXd& Y);

	void agetX(CovarInput* out) const;
	void setX(const CovarInput& X) throw (CGPMixException);

	inline muint_t getNumberSamples(){return this->cache->rgetYeffective().rows();} //get the number of training data samples
	inline muint_t getNumberDimension(){return this->cache->rgetYeffective().cols();} //get the dimension of the target data

	PGPCholCache getCache()
	{return this->cache;};

	PCovarianceFunction getCovar(){return covar;}
	PLikelihood getLik(){return lik;}
	PDataTerm getDataTerm() {return dataTerm;}


	//likelihood evaluation of current object
	virtual mfloat_t LML() throw (CGPMixException);
	//likelihood evaluation for new parameters
	virtual mfloat_t LML(const CGPHyperParams& params) throw (CGPMixException);
	//same for concatenated list of parameters
	virtual mfloat_t LML(const VectorXd& params) throw (CGPMixException);

	//overall gradient:
	virtual CGPHyperParams LMLgrad() throw (CGPMixException);
	virtual CGPHyperParams LMLgrad(const CGPHyperParams& params) throw (CGPMixException);
	virtual CGPHyperParams LMLgrad(const VectorXd& paramArray) throw (CGPMixException);

	virtual void aLMLgrad(VectorXd* out) throw (CGPMixException);
	virtual void aLMLgrad(VectorXd* out,const CGPHyperParams& params) throw (CGPMixException);
	virtual void aLMLgrad(VectorXd* out,const VectorXd& paramArray) throw (CGPMixException);

	//gradient components:
	virtual void aLMLgrad_covar(VectorXd* out) throw (CGPMixException);
	virtual void aLMLgrad_lik(VectorXd* out) throw (CGPMixException);
	virtual void aLMLgrad_X(MatrixXd* out) throw (CGPMixException);
	virtual void aLMLgrad_dataTerm(MatrixXd* out) throw (CGPMixException);
    
    //overall hessian
	virtual void aLMLhess(MatrixXd* out, stringVec vecLabels) throw (CGPMixException);
    
    //hessian components:
    virtual void aLMLhess_covar(MatrixXd* out) throw (CGPMixException);
    virtual void aLMLhess_lik(MatrixXd* out) throw (CGPMixException);
    virtual void aLMLhess_covarlik(MatrixXd* out) throw (CGPMixException);
    
    //laplace approximation stuff
    virtual void agetCov_laplace(MatrixXd* out, stringVec vecLabels) throw (CGPMixException);
    virtual CGPHyperParams agetStd_laplace() throw (CGPMixException);
    
	//interface for optimization:

	//predictions:
	virtual void apredictMean(MatrixXd* out, const MatrixXd& Xstar) throw (CGPMixException);
	virtual void apredictVar(MatrixXd* out, const MatrixXd& Xstar) throw (CGPMixException);

	//class factory for LMM instances:
	template <class lmmType>
	lmmType* getLMMInstance();

	//convenience function
	inline VectorXd LMLgrad_covar() throw (CGPMixException);
	inline VectorXd LMLgrad_lik() throw (CGPMixException);
	inline MatrixXd LMLgrad_X() throw (CGPMixException);
	inline MatrixXd LMLgrad_dataTerm() throw (CGPMixException);
	inline MatrixXd getY();
	inline MatrixXd getX() const;
	inline VectorXd getParamArray() const;
	inline MatrixXd predictMean(const MatrixXd& Xstar) throw (CGPMixException);
	inline MatrixXd predictVar(const MatrixXd& Xstar) throw (CGPMixException);
    
	/* Static methods*/
    //numerical gradient and hessian
    static double LMLgrad_num(CGPbase& gp, const muint_t i) throw(CGPMixException);
    static double LMLhess_num(CGPbase& gp, const muint_t i, const muint_t j) throw(CGPMixException);
    
};
typedef sptr<CGPbase> PGPbase;


inline MatrixXd CGPbase::predictMean(const MatrixXd& Xstar) throw (CGPMixException)
{
    MatrixXd rv;
    apredictMean(&rv,Xstar);
    return rv;
}

inline MatrixXd CGPbase::predictVar(const MatrixXd& Xstar)
		throw (CGPMixException)
		{
		MatrixXd rv;
		apredictVar(&rv,Xstar);
		return rv;
		}


inline MatrixXd CGPbase::getY()
{
	MatrixXd rv;
	this->agetY(&rv);
	return rv;
}

inline CovarInput CGPbase::getX() const
{
	MatrixXd rv;
	this->agetX(&rv);
	return rv;
}

inline VectorXd CGPbase::LMLgrad_covar() throw (CGPMixException)
{
	VectorXd rv;
	aLMLgrad_covar(&rv);
	return rv;
}


inline VectorXd CGPbase::LMLgrad_lik() throw (CGPMixException)
{
	VectorXd rv;
	aLMLgrad_lik(&rv);
	return rv;
}

inline MatrixXd CGPbase::LMLgrad_X() throw (CGPMixException)
{
	MatrixXd rv;
	aLMLgrad_X(&rv);
	return rv;
}

inline MatrixXd CGPbase::LMLgrad_dataTerm() throw (CGPMixException)
{
	MatrixXd rv;
	aLMLgrad_dataTerm(&rv);
	return rv;
}

inline VectorXd CGPbase::getParamArray() const
{
	VectorXd rv;
	agetParamArray(&rv);
	return rv;
}


//class factories for LMMs
template <class lmmType>
lmmType* CGPbase::getLMMInstance()
{
	//create instance
	lmmType* rv = new lmmType();
	//set K0
	MatrixXd& K0 = this->cache->getCovar()->rgetK();
	rv->setK(K0);
	//set phenotypes
	MatrixXd&  pheno = this->cache->rgetYeffective();
	rv->setPheno(pheno);
	return rv;
}


#if (defined(SWIG) && !defined(SWIG_FILE_WITH_INIT))
//%ignore CGPbase::getX;
//%rename(predictVar) CGPbase::apredictVar;
#endif

typedef std::vector<PGPbase> AGPbaseVec;
typedef std::vector<PLinearMean> ALinearMeanVec;

class CGPvarDecomp : public CGPbase {
protected:
	//GP vec
	AGPbaseVec vecGPs;

	//Dimensions
	muint_t P;
	muint_t N;
	muint_t state;

	//fixed effects
	//to be implemented
	//for now I only consider
	//a trait-specific intercept term
	MatrixXd fixed;
	//pheno
	MatrixXd pheno;

	//update params
	virtual void updateParams() throw (CGPMixException);

	//covariance stuff
	VectorXd lambda;
	ACovarVec C1;
	PTFixedCF C2;
	MatrixXd trait;
	ACovarVec covar;

	//dataterm
	ALinearMeanVec vecLinearMeans;

	//Initialization
	bool is_init;
	CovarParams initParams;

public:

	//CGPvarDecomp();
	CGPvarDecomp(PCovarianceFunction covar, PLikelihood lik,PDataTerm dataTerm,const VectorXd& lambda, const muint_t P, const MatrixXd& pheno, const VectorXd& initParams);
	virtual ~CGPvarDecomp();

	//Getters and setters
	//virtual void setParams(const CGPHyperParams& hyperparams) throw(CGPMixException);
	//PGPbase getGP(muint_t i){return *(vecGPs[i]);};
	//PLinearMean getMean(muint_t i){return vecLinearMeans[i];};
	//PCovarianceFunction getC1(muint_t i){return C1[i];};
	//PCovarianceFunction getC2(muint_t i){return C2;};
	muint_t getState(){return this->state;};
	void addGP(PGPbase gp)  throw (CGPMixException){vecGPs.push_back(gp);};
	void initializeGP(muint_t i,const CGPHyperParams& hyperparams){vecGPs[i]->setParams(hyperparams);};

	// Initialise vecGPs
	void initGPs() throw (CGPMixException);

	//LML
	virtual mfloat_t LML() throw (CGPMixException);
	//gradient components:
	virtual void aLMLgrad_covar(VectorXd* out) throw (CGPMixException);
	virtual void aLMLgrad_lik(VectorXd* out) throw (CGPMixException);
	virtual void aLMLgrad_X(MatrixXd* out) throw (CGPMixException);
	virtual void aLMLgrad_dataTerm(MatrixXd* out) throw (CGPMixException);

};

} /* namespace limix */
#endif /* GP_BASE_H_ */
