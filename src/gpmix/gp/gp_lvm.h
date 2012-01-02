/*
 * CGPlvm.h
 *
 *  Created on: Jan 2, 2012
 *      Author: stegle
 */

#ifndef CGPLVM_H_
#define CGPLVM_H_

#include "gp_base.h"

namespace gpmix {

class CGPlvm: public gpmix::CGPbase
{
protected:
	//index with entries in X which are used for GPLVM updates.
	VectorXi gplvmDimensions;
	virtual void updateParams() throw (CGPMixException);
public:
	CGPlvm(ACovarianceFunction& covar, ALikelihood& lik);
	virtual ~CGPlvm();

	void setX(const CovarInput& X) throw (CGPMixException);
	//additional gradients due to GPLVM components:
	CGPHyperParams LMLgrad() throw (CGPMixException);
	virtual void aLMLgrad_X(MatrixXd* out) throw (CGPMixException);
	inline MatrixXd LMLgrad_X() throw (CGPMixException);
};

inline MatrixXd CGPlvm::LMLgrad_X() throw (CGPMixException)
		{
		MatrixXd rv;
		aLMLgrad_X(&rv);
		return rv;
		}


} /* namespace gpmix */
#endif /* CGPLVM_H_ */
