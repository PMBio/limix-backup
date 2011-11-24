/*
 * fixed.h
 *
 *  Created on: Nov 11, 2011
 *      Author: stegle
 */

#ifndef FIXED_H_
#define FIXED_H_

#include <gpmix/covar/covariance.h>

namespace gpmix {

class CFixedCF : public ACovarianceFunction {
protected:
	MatrixXd K0;
public:
	CFixedCF(const MatrixXd K0);
	~CFixedCF();
	MatrixXd K(const CovarParams params, const CovarInput x1, const CovarInput x2) const;
	VectorXd Kdiag(const CovarParams params, const CovarInput x1) const;


	MatrixXd Kgrad_theta(const CovarParams params, const CovarInput x1, const muint_t i) const;
	MatrixXd Kgrad_x(const CovarParams params, const CovarInput x1, const CovarInput x2, const muint_t d) const;
	MatrixXd Kgrad_xdiag(const CovarParams params, const CovarInput x1, const muint_t d) const;
};

} /* namespace gpmix */
#endif /* FIXED_H_ */
