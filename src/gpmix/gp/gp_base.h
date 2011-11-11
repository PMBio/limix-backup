/*
 * gp_base.h
 *
 *  Created on: Nov 11, 2011
 *      Author: stegle
 */

#ifndef GP_BASE_H_
#define GP_BASE_H_

#include <gpmix/matrix/matrix_helper.h>
#include <gpmix/covar/covariance.h>
#include <gpmix/gp_types.h>

namespace gpmix {

class CGPbase {
public:
	CGPbase();
	virtual ~CGPbase();
};

} /* namespace gpmix */
#endif /* GP_BASE_H_ */
