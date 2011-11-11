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
	uint_t hyperparams;
	//indices for slicing out dimensions from X
	uint_t dimensions_i0;
	uint_t dimensions_i1;

	//project a full set of inputs onto the targets of this class:
	CovarInput getX(const CovarInput x) const;
	//is a particular dimension d within the targets of this class?
	bool dimension_is_target(const uint_t d) const;

public:
	ACovarianceFunction(const uint_t dimensions);
	ACovarianceFunction(const uint_t dimensions_i0,const uint_t dimensions_i1);
	virtual ~ACovarianceFunction();

	//TODO: add hyperparameters and think about a good construct for that
	virtual MatrixXd K(const CovarParams params, const CovarInput x1, const CovarInput x2) const =0;
	virtual VectorXd Kdiag(const CovarParams params, const CovarInput x1) const;

	virtual MatrixXd Kgrad_theta(const CovarParams params, const CovarInput x1, const uint_t i) const =0;
	virtual MatrixXd Kgrad_x(const CovarParams params, const CovarInput x1, const CovarInput x2, const uint_t d) const=0;
	virtual MatrixXd Kgrad_xdiag(const CovarParams params, const CovarInput x1, const uint_t d) const=0;
};

} /* namespace gpmix */
#endif /* ACOVARIANCE_H_ */
