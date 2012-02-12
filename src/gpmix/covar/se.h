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

#if (defined(SWIG) && !defined(SWIG_FILE_WITH_INIT))
//%shared_ptr(gpmix::CCovSqexpARD)
#endif
class CCovSqexpARD: public gpmix::ACovarianceFunction {
public:
	CCovSqexpARD(muint_t numberDimensions=1): ACovarianceFunction(1)
	{
		this->setNumberDimensions(numberDimensions);
	}

	~CCovSqexpARD();

	virtual void setNumberDimensions(muint_t numberDimensions);


	//overloaded pure virtual functions:
	virtual void aKcross(MatrixXd* out, const CovarInput& Xstar ) const throw(CGPMixException);
	virtual void aKcross_diag(VectorXd* out, const CovarInput& Xstar) const throw(CGPMixException);
	virtual void aKgrad_param(MatrixXd* out,const muint_t i) const throw(CGPMixException);
	virtual void aKcross_grad_X(MatrixXd* out,const CovarInput& Xstar, const muint_t d) const throw(CGPMixException);
	virtual void aKdiag_grad_X(VectorXd* out,const muint_t d) const throw(CGPMixException);

	//class information
	inline std::string getName() const{ return "CovSEARD";}
}; //end class CCovSqexpARD



} /* namespace gpmix */
#endif /* CCOVSQEXPARD_H_ */
