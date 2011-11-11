/*
 * Linear.h
 *
 *  Created on: Nov 11, 2011
 *      Author: stegle
 */

#ifndef LINEAR_H_
#define LINEAR_H_

#include <gpmix/matrix/matrix_helper.h>
#include <gpmix/covar/covariance.h>

namespace gpmix {

class CLinearCFISO: public ACovarianceFunction  {
public:
	CLinearCFISO();
	virtual ~CLinearCFISO();

	virtual MatrixXd K(CovarParams params,CovarInput x1,CovarInput x2);
	//virtual VectorXd Kdiag(CovarParams params,CovarInput x1);

	virtual MatrixXd Kgrad_theta(CovarParams params, CovarInput x1,int i);
	virtual MatrixXd Kgrad_x(CovarParams params,CovarInput x1,CovarInput x2,int d);
	virtual MatrixXd Kgrad_xdiag(CovarParams params,CovarInput x1,int d);
};

} /* namespace gpmix */
#endif /* LINEAR_H_ */
