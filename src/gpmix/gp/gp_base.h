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
#include <string>
#include <map>
#include <vector>
#include <gpmix/types.h>
using namespace std;

namespace gpmix {

//type of cholesky decomposition to use:
//LDL
//typedef Eigen::LDLT<gpmix::MatrixXd> MatrixXdChol;
//LL
typedef Eigen::LDLT<gpmix::MatrixXd> MatrixXdChol;



#if (defined(SWIG) && !defined(SWIG_FILE_WITH_INIT))
%ignore CGPHyperParams::get;
%ignore CGPHyperParams::getParamArray;

%rename(get) CGPHyperParams::aget;
%rename(getParamArray) CGPHyperParams::agetParamArray;
//PYTHON:
#ifdef SWIGPYTHON
%rename(__getitem__) CGPHyperParams::aget;
%rename(__setitem__) CGPHyperParams::set;
#endif

//list of strings: names of hyperparams
%template(StringVec) vector<string>;
//ok: this causes trouble at the moment, due to Eigen specifics.
//we rely on the interfaces we added for python use
//%template(StringMatrixMap) map<std::string,MatrixXd>;
#endif

typedef map<string,MatrixXd> CGPHyperParamsMap;

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

class CGPHyperParams : public map<string,MatrixXd> {

public:
	CGPHyperParams()
	{
	}
	//from a list of params
	~CGPHyperParams()
	{
	}

	void agetParamArray(VectorXd* out) const;
	void setParamArray(const VectorXd& param) throw (CGPMixException);

	muint_t getNumberParams() const;

	void set(const string& name, const MatrixXd& value);
	void aget(MatrixXd* out, const string& name);
	//get vector with existing names
	vector<string> getNames() const;
	//exists?
	bool exists(string name) const;

	//convenience functions for C++ access
	inline MatrixXd get(const string&name);
	inline VectorXd getParamArray();
};

inline MatrixXd CGPHyperParams::get(const string&name)
{
	MatrixXd rv;
	aget(&rv,name);
	return rv;
}
inline VectorXd CGPHyperParams::getParamArray()
{
	VectorXd rv;
	agetParamArray(&rv);
	return rv;
}



class CGPCache
{
public:

	MatrixXd K;
	MatrixXdChol cholK;
	MatrixXd Kinv;
	MatrixXd KinvY;
	MatrixXd DKinv_KinvYYKinv;
	CGPCache()
	{ clear();}

	void clear()
	{
		this->K=MatrixXd();
		this->Kinv=MatrixXd();
		this->KinvY=MatrixXd();
		this->cholK=MatrixXdChol();
		this->DKinv_KinvYYKinv = MatrixXd();
	}
};


#if (defined(SWIG) && !defined(SWIG_FILE_WITH_INIT))
%ignore CGPbase::getX;
%ignore CGPbase::getY;
%ignore CGPbase::LMLgrad_covar;
%ignore CGPbase::LMLgrad_lik;
%ignore CGPbase::getParamArray;

%rename(getParamArray) CGPbase::agetParamArray;
%rename(getX) CGPbase::agetX;
%rename(getY) CGPbase::agetY;
%rename(LMLgrad_covar) CGPbase::aLMLgrad_covar;
%rename(LMLgrad_lik) CGPbase::aLMLgrad_lik;
#endif


class CGPbase {
protected:

	MatrixXd Y;    //training targets
	//cached GP-parameters:
	CGPCache cache;
	CGPHyperParams params;

	ACovarianceFunction& covar;//Covariance function
	ALikelihood& lik;          //likelihood model

	virtual void clearCache();
	virtual bool isInSync() const;

	virtual MatrixXd* getK();
	virtual MatrixXd* getKinv();
	virtual MatrixXd* getKinvY();
	virtual MatrixXdChol* getCholK();
	virtual MatrixXd* getDKinv_KinvYYKinv();
	virtual void updateParams() throw (CGPMixException);

public:
	CGPbase(ACovarianceFunction& covar, ALikelihood& lik);
	virtual ~CGPbase();

	//TODO: add interface that is suitable for optimizer
	// virtual double LML(double* params);
	// virtual void LML(double* params, double* gradients);
	virtual void set_data(MatrixXd& Y);

	//getter and setter for Parameters:
	virtual void setParams(const CGPHyperParams& hyperparams) throw(CGPMixException);
	virtual CGPHyperParams getParams() const;
	virtual void setParamArray(const VectorXd& hyperparams) throw (CGPMixException);
	virtual void agetParamArray(VectorXd* out) const;


	void agetY(MatrixXd* out) const;
	void setY(const MatrixXd& Y);

	void agetX(CovarInput* out) const;
	void setX(const CovarInput& Y) throw (CGPMixException);

	inline muint_t getNumberSamples(){return this->Y.rows();} //get the number of training data samples
	inline muint_t getNumberDimension(){return this->Y.cols();} //get the dimension of the target data

	ACovarianceFunction* getCovar(){return &covar;}
	ALikelihood* getLik(){return &lik;}

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
	//interface for optimization:

	//predictions:
	virtual void apredictMean(MatrixXd* out, const MatrixXd& Xstar) throw (CGPMixException);
	virtual void apredictVar(MatrixXd* out, const MatrixXd& Xstar) throw (CGPMixException);

	//convenience function
	inline VectorXd LMLgrad_covar() throw (CGPMixException);
	inline VectorXd LMLgrad_lik() throw (CGPMixException);
	inline MatrixXd getY() const;
	inline MatrixXd getX() const;
	inline VectorXd getParamArray() const;
};


inline MatrixXd CGPbase::getY() const
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

inline VectorXd CGPbase::getParamArray() const
{
	VectorXd rv;
	agetParamArray(&rv);
	return rv;
}


} /* namespace gpmix */
#endif /* GP_BASE_H_ */
