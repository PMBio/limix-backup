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

	uint_t numberParams;
public:
	//constructors
#ifndef SWIG
	ACovarianceFunction(const uint_t numberParams);
#endif
	//destructors
	virtual ~ACovarianceFunction();
	//getters and setters

#ifndef SWIG
	//remove these members due to swig wrapping confusion of overladed function

	//computeK(X,X)
	virtual MatrixXd K() const;

	//compute K(Xstar,X)
	virtual MatrixXd Kcross( const CovarInput& Xstar ) const = 0;
	virtual VectorXd Kdiag() const;
	virtual MatrixXd K_grad_X(const uint_t d) const;
	virtual MatrixXd K_grad_param(const uint_t i) const = 0;

	//gradient of K(Xstar,X)
	virtual MatrixXd Kcross_grad_X(const CovarInput& Xstar, const uint_t d) const = 0;
	virtual MatrixXd Kdiag_grad_X(const uint_t d) const = 0;

	//class information
	virtual string getName() const = 0;

	//get the Vector of hyperparameters
	inline CovarParams getParams() const {return params;}

	//check if object is  insync with cache
	inline bool isInSync() const {return insync;}

	//indicate that the cache has been cleared and is synced again
	inline void makeSync() { insync = true;}

	//set the parameters to a new value.
	void setParams(CovarParams& params);

	//set X to a new value
	inline void setX(CovarInput& X);

	//get the X
	inline CovarInput getX() const;

	inline uint_t getDimX() const {return (uint_t)this->X.cols();}
#endif

	//getters and setters
	inline uint_t getNumberParams() const;



    /* Static methods
    //grad checking functions
    static bool check_covariance_Kgrad_theta(const ACovarianceFunction& covar,const CovarParams params, const CovarInput x,float_t relchange=1E-5,float_t threshold=1E-2);
    static bool check_covariance_Kgrad_x(const ACovarianceFunction& covar,const CovarParams params, const CovarInput x,float_t relchange=1E-5,float_t threshold=1E-2);
    //same, however auto generating random parameters:
    static bool check_covariance_Kgrad_theta(const ACovarianceFunction& covar,const uint_t n_rows = 100,float_t relchange=1E-5,float_t threshold=1E-2);
    static bool check_covariance_Kgrad_x(const ACovarianceFunction& covar,const uint_t n_rows = 100,float_t relchange=1E-5,float_t threshold=1E-2);
    */
};


	//set X to a new value
	inline void ACovarianceFunction::setX(CovarInput& X)
	{
		this->X = X;
		this->insync = false;
	}

	inline CovarInput ACovarianceFunction::getX() const
	{
		return this->X;
	}

	inline uint_t ACovarianceFunction::getNumberParams() const
	{
		return this->numberParams;
	}

//gradcheck tools for covaraince functions:

} /* namespace gpmix */


#endif /* ACOVARIANCE_H_ */
