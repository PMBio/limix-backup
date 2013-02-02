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

#include "limix/types.h"
namespace limix {

//define covariance function types

typedef MatrixXd CovarInput;
typedef VectorXd CovarParams;



//rename argout operators for swig interface
#if (defined(SWIG) && !defined(SWIG_FILE_WITH_INIT))
//ignore C++ versions
%ignore ACovarianceFunction::K;
%ignore ACovarianceFunction::Kdiag;
%ignore ACovarianceFunction::Kcross_diag;
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
%rename(Kcross_diag) ACovarianceFunction::aKcross_diag;
%rename(Kgrad_param) ACovarianceFunction::aKgrad_param;
%rename(Kcross_grad_X) ACovarianceFunction::aKcross_grad_X;

%rename(getParams) ACovarianceFunction::agetParams;
%rename(getX) ACovarianceFunction::agetX;
%rename(getParamBounds) ACovarianceFunction::agetParamBounds;
//%sptr(gpmix::ACovarianceFunction)
#endif

class ACovarianceFunction : public CParamObject {
protected:
	//the inputs of the kernel
	CovarInput X;
	//the hyperparameters of K
	CovarParams params;
	//mask for hyperparameter optimization
	CovarParams paramsMask;
	muint_t numberParams;
	muint_t numberDimensions;

	//helper functions:
	inline void checkWithinDimensions(muint_t d) const throw (CGPMixException);
	inline void checkWithinParams(muint_t i) const throw (CGPMixException);
	inline void checkXDimensions(const CovarInput& X) const throw (CGPMixException);
	inline void checkParamDimensions(const CovarParams& params) const throw (CGPMixException);
	void setNumberParams()
	{
		this->numberParams = numberParams;
	}

public:
	//constructors
	ACovarianceFunction(const muint_t numberParams=0);
	//destructors
	virtual ~ACovarianceFunction();

	//getters and setters
	virtual std::string getName() const = 0;

	//get the Vector of hyperparameters
	//set the parameters to a new value.
	virtual void setParams(const CovarParams& params);
	virtual void agetParams(CovarParams* out) const;
	//upper and lower constraint for hyper parameters
	virtual void agetParamBounds(CovarParams* lower,CovarParams* upper) const;
	virtual void agetParamMask(CovarParams* out) const;
	virtual void setParamMask(const CovarParams& params);
	virtual CovarParams getParamMask() const;


	//set X to a new value
	virtual void setX(const CovarInput& X) throw (CGPMixException);
	virtual void setXcol(const CovarInput& X, muint_t col) throw (CGPMixException);
	//get the X
	virtual void agetX(CovarInput* Xout) const throw (CGPMixException);
	virtual muint_t getDimX() const {return (muint_t)(this->X.cols());}
	virtual muint_t getNumberParams() const;
	virtual muint_t getNumberDimensions() const;
	virtual void setNumberDimensions(muint_t numberDimensions);

	// call by 
	//virtual functions that have trivial implementations
	virtual muint_t Kdim() const throw(CGPMixException);

	virtual void aK(MatrixXd* out) const;
	virtual void aKdiag(VectorXd* out) const;
	virtual void aKgrad_X(MatrixXd* out,const muint_t d) const throw(CGPMixException);

	//pure functions that need to be implemented
	virtual void aKcross(MatrixXd* out, const CovarInput& Xstar ) const throw(CGPMixException) = 0;
	virtual void aKcross_diag(VectorXd* out, const CovarInput& Xstar) const throw(CGPMixException) = 0;

	virtual void aKgrad_param(MatrixXd* out,const muint_t i) const throw(CGPMixException) =0;
	virtual void aKcross_grad_X(MatrixXd* out,const CovarInput& Xstar, const muint_t d) const throw(CGPMixException) = 0;
	virtual void aKdiag_grad_X(VectorXd* out,const muint_t d) const throw(CGPMixException) = 0;

	//Inline convenience functions:
	inline MatrixXd K() const;
	virtual CovarParams getParams() const;
	inline CovarInput getX() const;
	inline VectorXd Kdiag() const;
	inline MatrixXd Kcross( const CovarInput& Xstar ) const throw(CGPMixException);
	inline VectorXd Kcross_diag(const CovarInput& Xstar) const throw(CGPMixException);

	inline MatrixXd Kgrad_param(const muint_t i) const throw(CGPMixException);
	inline MatrixXd Kcross_grad_X(const CovarInput& Xstar, const muint_t d) const throw(CGPMixException);
	inline MatrixXd Kgrad_X(const muint_t d) const throw(CGPMixException);
	inline VectorXd Kdiag_grad_X(const muint_t d) const throw(CGPMixException);

	/* Static methods*/
	//grad checking functions
	static bool check_covariance_Kgrad_theta(ACovarianceFunction& covar,mfloat_t relchange=1E-5,mfloat_t threshold=1E-2);
	static bool check_covariance_Kgrad_x(ACovarianceFunction& covar,mfloat_t relchange=1E-5,mfloat_t threshold=1E-2,bool check_diag=true);
	//covariance normalization
	//template <typename Derived1, typename Derived2,typename Derived3,typename Derived4,typename Derived5>
	//inline static void scale_K(const Eigen::MatrixBase<Derived1> & K_);
};
typedef sptr<ACovarianceFunction> PCovarianceFunction;


/*
 * Cache object which inherits all the interface functions from Covar
 */
class CCovarianceFunctionCache : public CParamObject
{
protected:
	//wrapper covariance Function
	PCovarianceFunction covar;
	MatrixXd KCache; //K
	//SVD
	MatrixXd UCache; //U
	VectorXd SCache; //S
	//Chol
	MatrixXdChol cholKCache;
	bool KCacheNull,SVDCacheNull,cholKCacheNull;
	void updateSVD();
	void validateCache();
public:
	CCovarianceFunctionCache()
	{
	}

	CCovarianceFunctionCache(PCovarianceFunction covar)
	{
		setCovar(covar);
	}

	virtual ~CCovarianceFunctionCache()
	{
	};

	void setCovar(PCovarianceFunction covar);
	//transmit add and delete sync child to covar object
	virtual void addSyncChild(Pbool l)
	{
		covar->addSyncChild(l);
	}
	virtual void delSyncChild(Pbool l)
	{
		covar->delSyncChild(l);
	}

	//1. getter/setter
	inline PCovarianceFunction getCovar()
	{ return covar;}
	//2. cache interface
	virtual MatrixXd& rgetK();
	virtual MatrixXd& rgetUK();
	virtual VectorXd& rgetSK();
	virtual MatrixXdChol& rgetCholK();
};
typedef sptr<CCovarianceFunctionCache> PCovarianceFunctionCache;



/*Inline functions*/

/*
inline CovarParams ACovarianceFunction::getParams() const
{
	CovarParams rv;
	this->agetParams(&rv);
	return rv;
}
*/

inline MatrixXd ACovarianceFunction::getX() const
{
	MatrixXd rv;
	this->agetX(&rv);
	return rv;
}





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

inline VectorXd ACovarianceFunction::Kcross_diag(const CovarInput& Xstar) const throw (CGPMixException)
{
	VectorXd RV;
	aKcross_diag(&RV,Xstar);
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
		std::ostringstream os;
		os << "Dimension index ("<<d<<") out of range in covariance (0.."<<getNumberDimensions()<<").";
		throw CGPMixException(os.str());
	}
}
inline void ACovarianceFunction::checkWithinParams(muint_t i) const throw (CGPMixException)
{
	if (i>=getNumberParams())
	{
		std::ostringstream os;
		os << "Parameter index ("<<i<<") out of range in covariance (0.."<<getNumberParams()<<").";
		throw CGPMixException(os.str());
	}
}


inline void ACovarianceFunction::checkXDimensions(const CovarInput& X) const throw (CGPMixException)
{
	if ((muint_t)X.cols()!=this->getNumberDimensions())
	{
		std::ostringstream os;
		os << "X("<<(muint_t)X.rows()<<","<<(muint_t)X.cols()<<") column dimension missmatch (covariance: "<<this->getNumberDimensions() <<")";
		throw CGPMixException(os.str());
	}
}
inline void ACovarianceFunction::checkParamDimensions(const CovarParams& params) const throw (CGPMixException)
{
	if ((muint_t)(params.rows()) != this->getNumberParams()){
		std::ostringstream os;
		os << "Wrong number of params for covariance funtion " << this->getName() << ". numberParams = " << this->getNumberParams() << ", params.cols() = " << params.cols();
		throw CGPMixException(os.str());
	}
}

} /* namespace limix */


#endif /* ACOVARIANCE_H_ */
