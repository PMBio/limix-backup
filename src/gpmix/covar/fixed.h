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
	CFixedCF(MatrixXd K0);
	virtual ~CFixedCF();
};

} /* namespace gpmix */
#endif /* FIXED_H_ */
