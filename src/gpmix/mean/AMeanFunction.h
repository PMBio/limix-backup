/*
 * AMeanFunction.h
 *
 *  Created on: Jan 2, 2012
 *      Author: clippert
 */
#if 0
#ifndef AMEANFUNCTION_H_
#define AMEANFUNCTION_H_

#include <gpmix/types.h>

namespace gpmix {

typedef MatrixXd MeanInput;
typedef MatrixXd MeanParams;

class AMeanFunction {
	muint_t dimY;
	muint_t numberParams;
	MeanParams params;
	MeanInput fixedEffects;

public:
	AMeanFunction(const muint_t dimY = 1);
	virtual ~AMeanFunction();
	virtual MatrixXd grad_w() throw(CGPMixException) = 0;
	virtual MatrixXd f_w() throw(CGPMixException) = 0;
	virtual inline muint_t getNsamples(){return (muint_t)this->fixedEffects.rows();};
};

} /* namespace gpmix */
#endif /* AMEANFUNCTION_H_ */
#endif
