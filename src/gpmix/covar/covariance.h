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


class ACovarianceFunction {
protected:
	//indicator if the class is synced with the cache
	bool insync;

	//the inputs of the kernel
	CovarInput X;

	//the hyperparameters of K
	CovarParams params;

	muint_t numberParams;
public:
	//constructors
	ACovarianceFunction(const muint_t numberParams=0);
	//destructors
	virtual ~ACovarianceFunction();

	//getters and setters
	virtual string getName() const = 0;

	//get the Vector of hyperparameters
	inline void getParams(CovarParams* out){(*out) = params;};

	//check if object is  insync with cache
	inline bool isInSync() const {return insync;}
	//indicate that the cache has been cleared and is synced again
	inline void makeSync() { insync = true;}
	//set the parameters to a new value.
	virtual void setParams(const CovarParams& params);
	//set X to a new value
	inline virtual void setX(const CovarInput& X);
	//get the X
	inline void getX(CovarInput* Xout) const;
	inline muint_t getDimX() const {return (muint_t)(this->X.cols());
	}
	inline muint_t getNumberParams() const;

	//virtual functions that have trivial implementations
	virtual void K(MatrixXd* out) const;
	virtual void Kdiag(VectorXd* out) const;
	virtual void Kgrad_X(MatrixXd* out,const muint_t d) const;

	//pure functions that need to be implemented
	virtual void Kcross(MatrixXd* out, const CovarInput& Xstar ) const = 0;
	virtual void Kgrad_param(MatrixXd* out,const muint_t i) const =0;
	virtual void Kcross_grad_X(MatrixXd* out,const CovarInput& Xstar, const muint_t d) const = 0;
	virtual void Kdiag_grad_X(VectorXd* out,const muint_t d) const = 0;


#ifndef SWIG
	//Inline convenience functions:
	inline CovarParams getParams() const {return params;}
	inline CovarInput getX() const {return this->X;}
	inline virtual MatrixXd K() const;
	inline virtual VectorXd Kdiag() const;
	inline virtual MatrixXd Kcross( const CovarInput& Xstar ) const;
	inline virtual MatrixXd Kgrad_param(const muint_t i) const;
	inline virtual MatrixXd Kcross_grad_X(const CovarInput& Xstar, const muint_t d) const;
	inline virtual MatrixXd Kgrad_X(const muint_t d) const;
	inline virtual VectorXd Kdiag_grad_X(const muint_t d) const;
#endif

	/* Static methods*/
	//grad checking functions
	static bool check_covariance_Kgrad_theta(ACovarianceFunction& covar,mfloat_t relchange=1E-5,mfloat_t threshold=1E-2);
	static bool check_covariance_Kgrad_x(ACovarianceFunction& covar,mfloat_t relchange=1E-5,mfloat_t threshold=1E-2,bool check_diag=true);
};



/*Inline functions*/
inline void ACovarianceFunction::setX(const CovarInput & X)
{
	this->X = X;
	this->insync = false;
}

inline void ACovarianceFunction::getX(CovarInput *Xout) const
{
	(*Xout) = this->X;
}


inline muint_t ACovarianceFunction::getNumberParams() const
{
	return this->numberParams;
}



#ifndef SWIG

inline  MatrixXd ACovarianceFunction::K() const
{
	MatrixXd RV;
	K(&RV);
	return RV;
}


inline  VectorXd ACovarianceFunction::Kdiag() const
{
	VectorXd RV;
	Kdiag(&RV);
	return RV;
}
inline  MatrixXd ACovarianceFunction::Kcross( const CovarInput& Xstar ) const
{
	MatrixXd RV;
	Kcross(&RV,Xstar);
	return RV;
}



inline MatrixXd ACovarianceFunction::Kgrad_param(const muint_t i) const
{
	MatrixXd RV;
	Kgrad_param(&RV,i);
	return RV;
}

inline MatrixXd ACovarianceFunction::Kcross_grad_X(const CovarInput & Xstar, const muint_t d) const
{
	MatrixXd RV;
	Kcross_grad_X(&RV,Xstar,d);
	return RV;
}

inline MatrixXd ACovarianceFunction::Kgrad_X(const muint_t d) const
{
	MatrixXd RV;
	Kgrad_X(&RV,d);
	return RV;
}

inline VectorXd ACovarianceFunction::Kdiag_grad_X(const muint_t d) const
{
	VectorXd RV;
	Kdiag_grad_X(&RV,d);
	return RV;
}
#endif



//gradcheck tools for covaraince functions:

} /* namespace gpmix */


#endif /* ACOVARIANCE_H_ */
