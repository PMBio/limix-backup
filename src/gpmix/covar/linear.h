/*
 * Linear.h
 *
 *  Created on: Nov 11, 2011
 *      Author: stegle
 */

#ifndef LINEAR_H_
#define LINEAR_H_

#include <gpmix/covar/covariance.h>

namespace gpmix {

class CCovLinearISO: public ACovarianceFunction  {
public:
	CCovLinearISO(const uint_t dimensions) : ACovarianceFunction(dimensions)
	{
		//set number of hyperparams
		hyperparams = 1;
	}
	CCovLinearISO(const uint_t dimensions_i0,const uint_t dimensions_i1) : ACovarianceFunction(dimensions_i0,dimensions_i1) {
		//set number of hyperparams
		hyperparams = 1;
	}
	~CCovLinearISO();

	virtual string getName() const
		{ return "CovLinearISO";}


	MatrixXd K(const CovarParams params, const CovarInput x1, const CovarInput x2) const;
	MatrixXd Kgrad_theta(const CovarParams params, const CovarInput x1, const uint_t i) const;
	MatrixXd Kgrad_x(const CovarParams params, const CovarInput x1, const CovarInput x2, const uint_t d) const;
	MatrixXd Kgrad_xdiag(const CovarParams params, const CovarInput x1, const uint_t d) const;
};


class CCovLinearARD: public ACovarianceFunction  {
public:
	CCovLinearARD(const uint_t dimensions) : ACovarianceFunction(dimensions)
	{
		//set number of hyperparams: one parameter per data dimension
		hyperparams = dimensions;
	}
	CCovLinearARD(const uint_t dimensions_i0,const uint_t dimensions_i1) : ACovarianceFunction(dimensions_i0,dimensions_i1) {
		//set number of hyperparams
		hyperparams = dimensions_i1-dimensions_i0;
	}
	~CCovLinearARD();

	virtual string getName() const
	{ return "CovLinearARD";}

	MatrixXd K(const CovarParams params, const CovarInput x1, const CovarInput x2) const;
	MatrixXd Kgrad_theta(const CovarParams params, const CovarInput x1, const uint_t i) const;
	MatrixXd Kgrad_x(const CovarParams params, const CovarInput x1, const CovarInput x2, const uint_t d) const;
	MatrixXd Kgrad_xdiag(const CovarParams params, const CovarInput x1, const uint_t d) const;
};


} /* namespace gpmix */
#endif /* LINEAR_H_ */
