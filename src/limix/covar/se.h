// Copyright(c) 2014, The LIMIX developers (Christoph Lippert, Paolo Francesco Casale, Oliver Stegle)
// All rights reserved.
//
// LIMIX is provided under a 2-clause BSD license.
// See license.txt for the complete license.

#ifndef CCOVSQEXPARD_H_
#define CCOVSQEXPARD_H_

#include "covariance.h"

namespace limix {

class CCovSqexpARD: public ACovarianceFunction {
public:
	CCovSqexpARD(muint_t numberDimensions=1): ACovarianceFunction(1)
	{
		this->setNumberDimensions(numberDimensions);
		initParams();
	}

	~CCovSqexpARD();

	virtual void setNumberDimensions(muint_t numberDimensions);


	//overloaded pure virtual functions:
	virtual void aKcross(MatrixXd* out, const CovarInput& Xstar ) const ;
	virtual void aKcross_diag(VectorXd* out, const CovarInput& Xstar) const ;
	virtual void aKgrad_param(MatrixXd* out,const muint_t i) const ;
    virtual void aKhess_param(MatrixXd* out, const muint_t i, const muint_t j) const ;
	virtual void aKcross_grad_X(MatrixXd* out,const CovarInput& Xstar, const muint_t d) const ;
	virtual void aKdiag_grad_X(VectorXd* out,const muint_t d) const ;

	//class information
	inline std::string getName() const{ return "CovSEARD";}
}; //end class CCovSqexpARD

typedef sptr<CCovSqexpARD> PCovSqexpARD;


} /* namespace limix */
#endif /* CCOVSQEXPARD_H_ */
