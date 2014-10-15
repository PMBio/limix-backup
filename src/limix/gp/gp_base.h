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

#ifndef GP_BASE_H_
#define GP_BASE_H_

#include "limix/covar/covariance.h"
#include "limix/covar/combinators.h"
#include "limix/likelihood/likelihood.h"
#include "limix/mean/ADataTerm.h"
#include "limix/mean/CData.h"
#include "limix/types.h"
#include "limix/utils/cache.h"

#include <string>
#include <map>
#include <vector>
#include <iostream>


namespace limix {

/* Forward declaratins of classes */
class CGPbase;
class CGPKroneckerCache;
typedef sptr<CGPbase> PGPbase;

/*!
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
class CGPHyperParams : public std::map<std::string, MatrixXd>
{
protected:
	MatrixXd filterMask(const MatrixXd& array,const MatrixXd& mask) const;
	void expandMask(MatrixXd& out,const MatrixXd& array,const MatrixXd& mask) const;
	//	MatrixXd filterMask(const MatrixXd& array,const MatrixXd& mask) const;

public:
	CGPHyperParams()
	{
	}
	CGPHyperParams(const CGPHyperParams &_param); //< copy constructor

	~CGPHyperParams()
	{
	}

	void agetParamArray(VectorXd* out) const ;	//< returns the 1-D paramter vector without applying a mask
	void setParamArray(const VectorXd& param) ;	//< sets a 1-D paramter Vector without applying a mask

	void agetParamArray(VectorXd* out,const CGPHyperParams& mask) const ;		//< returns the 1-D paramter vector after applying a mask
	void setParamArray(const VectorXd& param,const CGPHyperParams& mask) ;	//< sets a 1-D paramter Vector after applying a mask

	muint_t getNumberParams() const; //!< returns the number of parameters without applying a mask.
	muint_t getNumberParams(const CGPHyperParams& mask) const; //!< returns the number of parameters after applying a mask.

	/**
	add a new parameter Mattrix to the paramters indexed by name
	@param	name	A string to index the parameter matrix
	@param	value	The parameter matrix
	*/
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
	inline MatrixXd get(const std::string& name);	//!< returns a specific set of parameters indexed by name.
	inline VectorXd getParamArray() const;			//!< returns the complete parameter vector
	inline VectorXd getParamArray(const CGPHyperParams& mask) const ;	//!< returns the parameter vector after applying the mask
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

inline VectorXd CGPHyperParams::getParamArray(const CGPHyperParams& mask) const 
{
	VectorXd rv;
	agetParamArray(&rv,mask);
	return rv;
}



//! Gaussian process caching class for Cholesky-basd inference
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
	//!> lik, covar and data term sync state
	Pbool syncLik,syncCovar,syncData;
	//TODO change this to shared pointer
	CGPbase* gp;
	PCovarianceFunctionCacheOld covar;
	//!> validate & clear cache
	void validateCache();
public:
	CGPCholCache(CGPbase* gp);	//!< constructor from a GP
	virtual ~CGPCholCache()	
	{};
	PCovarianceFunctionCacheOld getCovar()
	{
		return covar;
	}
	void setCovar(PCovarianceFunction covar);


	virtual MatrixXd& rgetKEff();			//!< returns the full covariance K
	virtual MatrixXdChol& rgetKEffChol();	//!< returns the Cholesky factorization of the full covariance K (K=LL'}
	virtual MatrixXd& rgetKEffInv();		//!< returns the inverse of the full covariance K	(K^{-1})
	virtual MatrixXd& rgetYeffective();		//!< returns the data term (with covariates)	(Y)
	virtual MatrixXd& rgetKEffInvY();		//!< returns the inverse covariance times the data term ( K^{-1}*Y )
	virtual MatrixXd& getDKEffInv_KEffInvYYKinv();	//!< returns a term required for the derivative of the covariance ( K^{-1}*YY'*K^{-1} )
};
typedef sptr<CGPCholCache> PGPCholCache;


class CGPbase : public enable_shared_from_this<CGPbase> {
	friend class CGPCholCache;
	friend class CGPKroneckerCache;
protected:

	//!> cached GP-parameters:
	PGPCholCache cache;
	CGPHyperParams params;

	//smart pointers for data Term,covar,lik
	PDataTerm dataTerm;       	//!< Mean function (smart pointer)
	PCovarianceFunction covar;	//!< Covariance function (smart pointer)
	PLikelihood lik;          	//!< likelihood model (smart pointer)

	VectorXi gplvmDimensions;	//!< gplvm dimensions (X for covar)

	virtual void updateParams() ;	//!< update the parameters of lik, dataterm, covar, X
	void updateX(ACovarianceFunction& covar,const VectorXi& gplvmDimensions,const MatrixXd& X) ;	//!< update covar X accorsing to gplvmDimensions

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

	//set penalization
	virtual void setLambda(mfloat_t lambda) {};
	virtual void setLambdaG(mfloat_t lambda) {};
	virtual void setLambdaN(mfloat_t lambda) {};

	//getter and setter for Parameters:
	virtual void setParams(const CGPHyperParams& hyperparams) ;					//!< sets the parameters without a mask
	virtual void setParams(const CGPHyperParams& hyperparams,const CGPHyperParams& mask) ;	//!< sets the parameters with a mask
	virtual CGPHyperParams getParams() const;															//!< returns all the parameters in the form of a CGPHyperparams object
	virtual void setParamArray(const VectorXd& hyperparams) ;					//!< sets the 1-D parameter vector without a mask
	virtual void setParamArray(const VectorXd& param,const CGPHyperParams& mask) ;	//!< sets the 1-D parameter vector with a mask
	virtual void agetParamArray(VectorXd* out) const;

	//getter for parameter bounds and hyperparam Mask
	virtual CGPHyperParams getParamBounds(bool upper) const;
	virtual CGPHyperParams getParamMask() const;

	void agetY(MatrixXd* out);
	void setY(const MatrixXd& Y);

	void agetX(CovarInput* out) const;
	void setX(const CovarInput& X) ;

	inline muint_t getNumberSamples(){return this->cache->rgetYeffective().rows();} //!< get the number of training data samples
	inline muint_t getNumberDimension(){return this->cache->rgetYeffective().cols();} //!< get the dimension of the target data

	PGPCholCache getCache()
	{return this->cache;};

	PCovarianceFunction getCovar(){return covar;}
	PLikelihood getLik(){return lik;}
	PDataTerm getDataTerm() {return dataTerm;}

	//get from cache
	virtual void agetKEffInvYCache(MatrixXd* out) ;

	//!> likelihood evaluation of current object
	virtual mfloat_t LML() ;
	//!> likelihood evaluation for new parameters
	virtual mfloat_t LML(const CGPHyperParams& params) ;
	//!> likelihood evaluationfor concatenated list (1-D vector) of parameters
	virtual mfloat_t LML(const VectorXd& params) ;

	//!>overall gradient:
	virtual CGPHyperParams LMLgrad() ;
	virtual CGPHyperParams LMLgrad(const CGPHyperParams& params) ;	//!< overall gradient for parameter object
	virtual CGPHyperParams LMLgrad(const VectorXd& paramArray) ;		//!< overall gradient for parameter 1-D vector

	//!>overall gradient:
	virtual void aLMLgrad(VectorXd* out) ;
	virtual void aLMLgrad(VectorXd* out,const CGPHyperParams& params) ;	//!< overall gradient for parameter object
	virtual void aLMLgrad(VectorXd* out,const VectorXd& paramArray) ;	//!< overall gradient for parameter 1-D vector

	//gradient components:
	virtual void aLMLgrad_covar(VectorXd* out) ;		//!< gradient component: gradient of covariance
	virtual void aLMLgrad_lik(VectorXd* out) ;		//!< gradient component: gradient of likelihood term
	virtual void aLMLgrad_X(MatrixXd* out) ;			//!< gradient component: gradient of X (GPLVM dimensions)
	virtual void aLMLgrad_dataTerm(MatrixXd* out) ;	//!< gradient component: gradient of the data term
    
    //!> overall hessian
	virtual void aLMLhess(MatrixXd* out, stringVec vecLabels) ;
    
    //hessian components:
    virtual void aLMLhess_covar(MatrixXd* out) ;		//!< Hessian component: for covariance
    virtual void aLMLhess_lik(MatrixXd* out) ;		//!< Hessian component: for likelihood term
    virtual void aLMLhess_covarlik(MatrixXd* out) ;	//!< Hessian component: between covariance and likelihood parameters
    
    //laplace approximation stuff
    virtual void agetCov_laplace(MatrixXd* out, stringVec vecLabels) ;	//!< Laplace approximation stuff: Inverse HEssian
    virtual CGPHyperParams agetStd_laplace() ;	//!< Laplace approximation stuff: standard deviation of parameters (sqrt of diagonal of inverse Hessian)
    
	//interface for optimization:

	//predictions:
	virtual void apredictMean(MatrixXd* out, const MatrixXd& Xstar) ;	//!< Conditional mean prediction at covariance input parameters Xstar, a.k.a Best Linear Unbiased Prediction (BLUP)
	virtual void apredictVar(MatrixXd* out, const MatrixXd& Xstar) ;		//!< Predictive variances around the BLUP. Note that Variances are coomputed independently (no co-variances)

	//!>class factory for LMM instances:
	template <class lmmType>
	lmmType* getLMMInstance();	//!< creates an LMM object for GWAS testing

	//convenience function
	inline VectorXd LMLgrad_covar() ;		//!< gradient component: gradient of covariance
	inline VectorXd LMLgrad_lik() ;			//!< gradient component: gradient of likelihood term
	inline MatrixXd LMLgrad_X() ;			//!< gradient component: gradient of X (GPLVM dimensions)
	inline MatrixXd LMLgrad_dataTerm() ;		//!< gradient component: gradient of the data term
	inline MatrixXd getY();									//!< returns Y
	inline MatrixXd getX() const;							//!< returns X (the covariance inputs)
	inline VectorXd getParamArray() const;					//!< returns the parameters as 1-D vector
	inline MatrixXd predictMean(const MatrixXd& Xstar) ;	//!< Conditional mean prediction at covariance input parameters Xstar, a.k.a Best Linear Unbiased Prediction (BLUP)
	inline MatrixXd predictVar(const MatrixXd& Xstar) ;	//!< Predictive variances around the BLUP. Note that Variances are coomputed independently (no co-variances)
    
	/* Static methods*/
    //numerical gradient and hessian
    static double LMLgrad_num(CGPbase& gp, const muint_t i) ;						//!< numerical evaluation of gradient for parameter i in the 1-D vector (slow! for debuging purposes)
    static double LMLhess_num(CGPbase& gp, const muint_t i, const muint_t j) ;	//!< numerical evaluation of Hessian between parameters i and j in the 1-D vector (slow! for debuging purposes)
    
};
typedef sptr<CGPbase> PGPbase;


inline MatrixXd CGPbase::predictMean(const MatrixXd& Xstar) 
{
    MatrixXd rv;
    apredictMean(&rv,Xstar);
    return rv;
}

inline MatrixXd CGPbase::predictVar(const MatrixXd& Xstar)
		
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

inline VectorXd CGPbase::LMLgrad_covar() 
{
	VectorXd rv;
	aLMLgrad_covar(&rv);
	return rv;
}


inline VectorXd CGPbase::LMLgrad_lik() 
{
	VectorXd rv;
	aLMLgrad_lik(&rv);
	return rv;
}

inline MatrixXd CGPbase::LMLgrad_X() 
{
	MatrixXd rv;
	aLMLgrad_X(&rv);
	return rv;
}

inline MatrixXd CGPbase::LMLgrad_dataTerm() 
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
	// create LMM instance
	lmmType* rv = new lmmType();
	// set K0
	MatrixXd& K0 = this->cache->getCovar()->rgetK();
	rv->setK(K0);
	//set phenotypes
	MatrixXd&  pheno = this->cache->rgetYeffective();
	rv->setPheno(pheno);
	return rv;
}


} /* namespace limix */
#endif /* GP_BASE_H_ */
