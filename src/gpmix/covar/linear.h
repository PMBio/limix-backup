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
	CCovLinearISO(const uint_t dimensions) : ACovarianceFunction(dimensions) {}
	CCovLinearISO(const uint_t dimensions_i0,const uint_t dimensions_i1) : ACovarianceFunction(dimensions_i0,dimensions_i1) {}
	~CCovLinearISO();

	MatrixXd K(const CovarParams params, const CovarInput x1, const CovarInput x2) const;
	MatrixXd Kgrad_theta(const CovarParams params, const CovarInput x1, const uint_t i) const;
	MatrixXd Kgrad_x(const CovarParams params, const CovarInput x1, const CovarInput x2, const uint_t d) const;
	MatrixXd Kgrad_xdiag(const CovarParams params, const CovarInput x1, const uint_t d) const;
};

} /* namespace gpmix */
#endif /* LINEAR_H_ */
