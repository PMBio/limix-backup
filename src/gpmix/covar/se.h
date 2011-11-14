/*
 * CCovSqexpARD.h
 *
 *  Created on: Nov 11, 2011
 *      Author: stegle
 */

#ifndef CCOVSQEXPARD_H_
#define CCOVSQEXPARD_H_

#include "covariance.h"

namespace gpmix {

class CCovSqexpARD: public gpmix::ACovarianceFunction {
public:
	CCovSqexpARD(const uint_t dimensions): ACovarianceFunction(dimensions)
	{
		hyperparams = 1+dimensions;
	}
	CCovSqexpARD(const uint_t dimensions_i0,const uint_t dimensions_i1) : ACovarianceFunction(dimensions_i0,dimensions_i1)
	{
		hyperparams = 1+(dimensions_i1-dimensions_i0);
	}
	MatrixXd K(const CovarParams params, const CovarInput x1, const CovarInput x2) const;
	VectorXd Kdiag(const CovarParams params, const CovarInput x1) const;
	MatrixXd Kgrad_theta(const CovarParams params, const CovarInput x1, const uint_t i) const;
	MatrixXd Kgrad_x(const CovarParams params, const CovarInput x1, const CovarInput x2, const uint_t d) const;
	MatrixXd Kgrad_xdiag(const CovarParams params, const CovarInput x1, const uint_t d) const;

	virtual ~CCovSqexpARD();
};

} /* namespace gpmix */
#endif /* CCOVSQEXPARD_H_ */
