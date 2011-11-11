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

#include <gpmix/matrix/matrix_helper.h>
#include <gpmix/gp_types.h>

namespace gpmix {

//define covariance function types

typedef MatrixXd CovarInput;
typedef MatrixXd CovarParams;


class ACovarianceFunction {
protected:
	unsigned int hyperparams;
	unsigned int dimensions;
	//indices for slicing out dimensions from X
	unsigned int dimensions_i0;
	unsigned int dimensions_i1;

	//project a full set of inputs onto the targets of this class:
	CovarInput getX(CovarInput x);
	//is a particular dimension d within the targets of this class?
	bool dimension_is_target(const unsigned int d);

public:
	ACovarianceFunction(const unsigned int dimensions=0, const unsigned int dimensions_i0=0, const unsigned int dimensions_i1=0);
	virtual ~ACovarianceFunction();

	//TODO: add hyperparameters and think about a good construct for that
	virtual MatrixXd K(const CovarParams params, const CovarInput x1, const CovarInput x2) const =0;
	virtual VectorXd Kdiag(const CovarParams params, const CovarInput x1);

	virtual MatrixXd Kgrad_theta(const CovarParams params, const CovarInput x1, const unsigned int i) const =0;
	virtual MatrixXd Kgrad_x(const CovarParams params, const CovarInput x1, const CovarInput x2, const unsigned int d) const=0;
	virtual MatrixXd Kgrad_xdiag(const CovarParams params, const CovarInput x1, const unsigned int d) const=0;
};

} /* namespace gpmix */
#endif /* ACOVARIANCE_H_ */
