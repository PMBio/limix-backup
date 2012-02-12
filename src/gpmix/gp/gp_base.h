/*
 * gp_base.h
 *
 *  Created on: Nov 11, 2011
 *      Author: stegle
 */

#ifndef GP_BASE_H_
#define GP_BASE_H_

#include <gpmix/covar/covariance.h>
#include <gpmix/likelihood/likelihood.h>
#include <gpmix/mean/ADataTerm.h>
#include <gpmix/mean/CData.h>
#include <string>
#include <map>
#include <vector>
#include <gpmix/types.h>
#include <iostream>


namespace gpmix {

/* Forward declaratins of classes */
class CGPbase;
class CGPKroneckerCache;
typedef sptr<CGPbase> PGPbase;



//type of cholesky decomposition to use:
//LDL
//typedef Eigen::LDLT<gpmix::MatrixXd> MatrixXdChol;
//LL
//typedef Eigen::LDLT<gpmix::MatrixXd> MatrixXdChol;
typedef Eigen::LLT<gpmix::MatrixXd> MatrixXdChol;



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
#endif

//TODO: work on map handling in swig
//%template(StringMatrixMap) map<std::string,MatrixXd>;
//%shared_ptr(gpmix::CGPHyperParams)
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



//cache class for a covariance function.
//offers cached access to a number of covaraince accessors and derived quantities:
class CGPCholCache
{
protected:
	MatrixXd K;
	MatrixXd K0;
	MatrixXdChol cholK;
	MatrixXd Kinv;
	MatrixXd KinvY;
	MatrixXd DKinv_KinvYYKinv;
	MatrixXd Yeffective;
	bool KNull,K0Null,cholKNull,KinvNull,KinvYNull,DKinv_KinvYYKinvNull,YeffectiveNull,gradDataParamsNull,gradDataParamsColsNull;
	CGPbase* gp;
	PCovarianceFunction covar;
public:
	CGPCholCache(CGPbase* gp,PCovarianceFunction covar);
	virtual ~CGPCholCache()
	{};

	void setCovar(PCovarianceFunction covar);
	virtual void clearCache();
	virtual bool isInSync() const;

	MatrixXd& getK0();
	MatrixXd& getK();
	MatrixXd& getKinv();
	MatrixXd& getYeffective();
	virtual MatrixXd& getKinvY();
	MatrixXdChol& getCholK();
	MatrixXd& getDKinv_KinvYYKinv();

	void agetK0(MatrixXd* out)
	{
		(*out) =  getK0();
	}
	void agetK(MatrixXd* out)
	{
		(*out) =  getK();
	}
};


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
%rename(predictMean) CGPbase::apredictMean;
%rename(predictVar) CGPbase::apredictVar;
//
//%shared_ptr(gpmix::CGPbase)
#endif

class CGPbase : public enable_shared_from_this<CGPbase> {
	friend class CGPCholCache;
	friend class CGPKroneckerCache;
protected:

	//cached GP-parameters:
	CGPCholCache cache;
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


	void agetY(MatrixXd* out);
	void setY(const MatrixXd& Y);

	void agetX(CovarInput* out) const;
	void setX(const CovarInput& X) throw (CGPMixException);

	inline muint_t getNumberSamples(){return this->cache.getYeffective().rows();} //get the number of training data samples
	inline muint_t getNumberDimension(){return this->cache.getYeffective().cols();} //get the dimension of the target data


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
};


inline MatrixXd CGPbase::predictMean(const MatrixXd& Xstar) throw (CGPMixException)
		{
		MatrixXd rv;
		apredictMean(&rv,Xstar);
		return rv;
		}
inline MatrixXd CGPbase::predictVar(const MatrixXd& Xstar) throw (CGPMixException)
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
	MatrixXd& K0 = this->cache.getK0();
	rv->setK(K0);
	//set phenotypes
	MatrixXd&  pheno = this->cache.getYeffective();
	rv->setPheno(pheno);
	return rv;
}

} /* namespace gpmix */
#endif /* GP_BASE_H_ */
