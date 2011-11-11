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
	int hyperparams;
	int dimensions;
	//indices for slicing out dimensions from X
	int dimensions_i0;
	int dimensions_i1;

	//project a fullset of inputs onto the targets of this class:
	CovarInput getX(CovarInput x);
	//is a particular dimension d wihtin the targets of this class?
	bool dimension_is_target(int d);

public:
	ACovarianceFunction(int dimensions=-1,int dimensions_i0=-1,int dimensions_i1=-1);
	virtual ~ACovarianceFunction();

	//TODO: add hyperparameters and think about a good construct for that
	virtual MatrixXd K(CovarParams params,CovarInput x1,CovarInput x2) const =0;
	virtual VectorXd Kdiag(CovarParams params,CovarInput x1);

	virtual MatrixXd Kgrad_theta(CovarParams params, CovarInput x1,int i) const =0;
	virtual MatrixXd Kgrad_x(CovarParams params,CovarInput x1,CovarInput x2,int d) const=0;
	virtual MatrixXd Kgrad_xdiag(CovarParams params,CovarInput x1,int d) const=0;



};

} /* namespace gpmix */
#endif /* ACOVARIANCE_H_ */
