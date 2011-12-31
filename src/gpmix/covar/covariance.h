/*
 * ACovariance.h
 *
 * Abstract definition of a covaraince function
 *
 *  Created on: Nov 10, 2011
 *      Author: stegle
 */

#ifndef ACOVARIANCE_H_
#define ACOVARIANCE_H_

#include <gpmix/types.h>
namespace gpmix {

//define covariance function types

typedef MatrixXd CovarInput;
typedef VectorXd CovarParams;



//rename argout operators for swig interface
#if (defined(SWIG) && !defined(SWIG_FILE_WITH_INIT))
//ignore C++ versions
%ignore ACovarianceFunction::K;
%ignore ACovarianceFunction::Kdiag;
%ignore ACovarianceFunction::Kdiag_grad_X;
%ignore ACovarianceFunction::Kgrad_X;
%ignore ACovarianceFunction::Kcross;
%ignore ACovarianceFunction::Kgrad_param;
%ignore ACovarianceFunction::Kcross_grad_X;

%ignore ACovarianceFunction::getParams;
%ignore ACovarianceFunction::getX;

//rename argout versions for python; this overwrites the C++ convenience functions
%rename(K) ACovarianceFunction::aK;
%rename(Kdiag) ACovarianceFunction::aKdiag;
%rename(Kdiag_grad_X) ACovarianceFunction::aKdiag_grad_X;
%rename(Kgrad_X) ACovarianceFunction::aKgrad_X;
%rename(Kcross) ACovarianceFunction::aKcross;
%rename(Kgrad_param) ACovarianceFunction::aKgrad_param;
%rename(Kcross_grad_X) ACovarianceFunction::aKcross_grad_X;

%rename(getParams) ACovarianceFunction::agetParams;
%rename(getX) ACovarianceFunction::agetX;
#endif

class ACovarianceFunction {
protected:
	//indicator if the class is synced with the cache
	bool insync;
	//the inputs of the kernel
	CovarInput X;
	//the hyperparameters of K
	CovarParams params;
	muint_t numberParams;
	muint_t numberDimensions;
	//helper functions:
	inline void checkWithinDimensions(muint_t d) const throw (CGPMixException);
	inline void checkWithinParams(muint_t i) const throw (CGPMixException);
	inline void checkXDimensions(const CovarInput& X) const throw (CGPMixException);
	inline void checkParamDimensions(const CovarParams& params) const throw (CGPMixException);
public:
	//constructors
	ACovarianceFunction(const muint_t numberParams=0);
	//destructors
	virtual ~ACovarianceFunction();

	//getters and setters
	virtual string getName() const = 0;

	//get the Vector of hyperparameters
	//set the parameters to a new value.
	virtual void setParams(const CovarParams& params);
	virtual void agetParams(CovarParams* out);

	//check if object is  insync with cache
	virtual bool isInSync() const;
	//indicate that the cache has been cleared and is synced again
	virtual void makeSync();
	//set X to a new value
	virtual void setX(const CovarInput& X) throw (CGPMixException);
	//get the X
	virtual void agetX(CovarInput* Xout) const throw (CGPMixException);
	inline muint_t getDimX() const {return (muint_t)(this->X.cols());
	}
	virtual muint_t getNumberParams() const;
	virtual muint_t getNumberDimensions() const;
	virtual void setNumberDimensions(muint_t numberDimensions);

	// call by 
	//virtual functions that have trivial implementations
	virtual void aK(MatrixXd* out) const;
	virtual void aKdiag(VectorXd* out) const;
	virtual void aKgrad_X(MatrixXd* out,const muint_t d) const throw(CGPMixException);

	//pure functions that need to be implemented
	virtual void aKcross(MatrixXd* out, const CovarInput& Xstar ) const throw(CGPMixException) = 0;
	virtual void aKgrad_param(MatrixXd* out,const muint_t i) const throw(CGPMixException) =0;
	virtual void aKcross_grad_X(MatrixXd* out,const CovarInput& Xstar, const muint_t d) const throw(CGPMixException) = 0;
	virtual void aKdiag_grad_X(VectorXd* out,const muint_t d) const throw(CGPMixException) = 0;

	//Inline convenience functions:
	inline MatrixXd K() const;
	inline CovarParams getParams() const {return params;}
	inline CovarInput getX() const {return this->X;}
	inline VectorXd Kdiag() const;
	inline MatrixXd Kcross( const CovarInput& Xstar ) const throw(CGPMixException);
	inline MatrixXd Kgrad_param(const muint_t i) const throw(CGPMixException);
	inline MatrixXd Kcross_grad_X(const CovarInput& Xstar, const muint_t d) const throw(CGPMixException);
	inline MatrixXd Kgrad_X(const muint_t d) const throw(CGPMixException);
	inline VectorXd Kdiag_grad_X(const muint_t d) const throw(CGPMixException);

	/* Static methods*/
	//grad checking functions
	static bool check_covariance_Kgrad_theta(ACovarianceFunction& covar,mfloat_t relchange=1E-5,mfloat_t threshold=1E-2);
	static bool check_covariance_Kgrad_x(ACovarianceFunction& covar,mfloat_t relchange=1E-5,mfloat_t threshold=1E-2,bool check_diag=true);
};



/*Inline functions*/


inline  MatrixXd ACovarianceFunction::K() const
{
	MatrixXd RV;
	aK(&RV);
	return RV;
}

inline  VectorXd ACovarianceFunction::Kdiag() const
{
	VectorXd RV;
	aKdiag(&RV);
	return RV;
}

inline  MatrixXd ACovarianceFunction::Kcross( const CovarInput& Xstar ) const throw(CGPMixException)
{
	MatrixXd RV;
	aKcross(&RV,Xstar);
	return RV;
}

inline MatrixXd ACovarianceFunction::Kgrad_param(const muint_t i) const throw(CGPMixException)
{
	MatrixXd RV;
	aKgrad_param(&RV,i);
	return RV;
}

inline MatrixXd ACovarianceFunction::Kcross_grad_X(const CovarInput & Xstar, const muint_t d) const throw(CGPMixException)
{
	MatrixXd RV;
	aKcross_grad_X(&RV,Xstar,d);
	return RV;
}

inline MatrixXd ACovarianceFunction::Kgrad_X(const muint_t d) const throw(CGPMixException)
{
	MatrixXd RV;
	aKgrad_X(&RV,d);
	return RV;
}

inline VectorXd ACovarianceFunction::Kdiag_grad_X(const muint_t d) const throw(CGPMixException)
{
	VectorXd RV;
	aKdiag_grad_X(&RV,d);
	return RV;
}




inline void ACovarianceFunction::checkWithinDimensions(muint_t d) const throw (CGPMixException)
{
	if (d>=getNumberDimensions())
	{
		ostringstream os;
		os << "Dimension index ("<<d<<") out of range in covariance (0.."<<getNumberDimensions()<<").";
		throw CGPMixException(os.str());
	}
}
inline void ACovarianceFunction::checkWithinParams(muint_t i) const throw (CGPMixException)
{
	if (i>=getNumberParams())
	{
		ostringstream os;
		os << "Parameter index ("<<i<<") out of range in covariance (0.."<<getNumberParams()<<").";
		throw CGPMixException(os.str());
	}
}


inline void ACovarianceFunction::checkXDimensions(const CovarInput& X) const throw (CGPMixException)
{
	if ((muint_t)X.cols()!=this->getNumberDimensions())
	{
		ostringstream os;
		os << "X("<<(muint_t)X.rows()<<","<<(muint_t)X.cols()<<") column dimension missmatch (covariance: "<<this->getNumberDimensions() <<")";
		throw CGPMixException(os.str());
	}
}
inline void ACovarianceFunction::checkParamDimensions(const CovarParams& params) const throw (CGPMixException)
{
	if ((muint_t)(params.rows()) != this->getNumberParams()){
		ostringstream os;
		os << "Wrong number of params for covariance funtion " << this->getName() << ". numberParams = " << this->getNumberParams() << ", params.cols() = " << params.cols();
		throw gpmix::CGPMixException(os.str());
	}
}



//gradcheck tools for covaraince functions:

} /* namespace gpmix */


#endif /* ACOVARIANCE_H_ */
