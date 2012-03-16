/*
 * freeform.h
 *
 *  Created on: Jan 16, 2012
 *      Author: stegle
 */

#ifndef FREEFORM_H_
#define FREEFORM_H_

#include "covariance.h"

namespace limix {

#if (defined(SWIG) && !defined(SWIG_FILE_WITH_INIT))
//%shared_ptr(gpmix::CCovFreeform)
#endif
class CCovFreeform: public ACovarianceFunction {
protected:
	muint_t numberGroups;
	void agetL0grad_param(MatrixXd* out,muint_t i) const throw(CGPMixException);
	void agetK0grad_param(MatrixXd* out,muint_t i) const throw(CGPMixException);
	void projectKcross(MatrixXd* out,const MatrixXd& K0,const CovarInput& Xstar) const throw (CGPMixException);
public:
	CCovFreeform(muint_t numberGroups);
	virtual ~CCovFreeform();

	void agetL0(MatrixXd* out) const;
	void agetK0(MatrixXd* out) const;


	void aKcross(MatrixXd* out, const CovarInput& Xstar ) const throw(CGPMixException);
	void aKcross_diag(VectorXd* out, const CovarInput& Xstar) const throw(CGPMixException);

	void aKgrad_param(MatrixXd* out,const muint_t i) const throw(CGPMixException);
	void aKcross_grad_X(MatrixXd* out,const CovarInput& Xstar, const muint_t d) const throw(CGPMixException);
	void aKdiag_grad_X(VectorXd* out,const muint_t d) const throw(CGPMixException);

	//class information
	inline std::string getName() const {return "CFreeform";};

};

} /* namespace limix */
#endif /* FREEFORM_H_ */
