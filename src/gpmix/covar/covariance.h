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
typedef MatrixXd CovarParams;


class ACovarianceFunction {
protected:
	//number of hyperparameters
	uint_t hyperparams;
	//number of dimensions
	uint_t dimensions;
	//indices for slicing out dimensions from X
	uint_t dimensions_i0;
	uint_t dimensions_i1;

	//project a full set of inputs onto the targets of this class:
	CovarInput getX(const CovarInput x) const;
	//is a particular dimension d within the targets of this class?
	bool dimension_is_target(const uint_t d) const;

public:
	//constructors
#ifndef SWIG
	ACovarianceFunction(const uint_t dimensions);
#endif
	ACovarianceFunction(const uint_t dimensions_i0,const uint_t dimensions_i1);
	//destructors
	virtual ~ACovarianceFunction();
	//getters and setters




#ifndef SWIG
	//remove these members due to swig wrapping confusion of overladed function
	virtual MatrixXd K(const CovarParams params, const CovarInput x1) const;
	virtual MatrixXd K(const CovarParams params, const CovarInput x1, const CovarInput x2) const =0;
	virtual VectorXd Kdiag(const CovarParams params, const CovarInput x1) const;

	virtual MatrixXd Kgrad_x(const CovarParams params, const CovarInput x1, const uint_t d) const;
	virtual MatrixXd Kgrad_theta(const CovarParams params, const CovarInput x1, const uint_t i) const =0;
	virtual MatrixXd Kgrad_x(const CovarParams params, const CovarInput x1, const CovarInput x2, const uint_t d) const=0;
	virtual MatrixXd Kgrad_xdiag(const CovarParams params, const CovarInput x1, const uint_t d) const=0;
	//class information
	virtual string getName() const = 0;
#endif


	//getters and setters
	uint_t getDimensions() const;
    uint_t getDimensionsI0() const;
    uint_t getDimensionsI1() const;
    uint_t getHyperparams() const;

    /* Static methods */
    //grad checking functions
    static bool check_covariance_Kgrad_theta(const ACovarianceFunction& covar,const CovarParams params, const CovarInput x,float_t relchange=1E-5,float_t threshold=1E-2);
    static bool check_covariance_Kgrad_x(const ACovarianceFunction& covar,const CovarParams params, const CovarInput x,float_t relchange=1E-5,float_t threshold=1E-2);
    //same, however auto generating random parameters:
    static bool check_covariance_Kgrad_theta(const ACovarianceFunction& covar,const uint_t n_rows = 100,float_t relchange=1E-5,float_t threshold=1E-2);
    static bool check_covariance_Kgrad_x(const ACovarianceFunction& covar,const uint_t n_rows = 100,float_t relchange=1E-5,float_t threshold=1E-2);
};



//gradcheck tools for covaraince functions:

} /* namespace gpmix */


#endif /* ACOVARIANCE_H_ */
